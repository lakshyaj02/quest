import math
import torch
from einops import rearrange
import torch.nn as nn

import pdb

from .utils import exact_attention, exact_attention_cuda, add_self_attentions, indexing, get_distribution
from .angular_lsh import AngularLSH, LSH

from torch import Tensor, nn

from .kv_cache import KVCache, DoubleKVCache, SingleKVCache
from dataclasses import dataclass
from typing import Literal

import torch.nn.functional as F

from open_lm.attention import torch_attn, xformers_attn

class HyperAttention(torch.nn.Module):

    def __init__(self, input_dim=64, lsh_num_projs=7, block_size=256, sample_size=256, min_seq_len=4096, cuda=False):
        super().__init__()
        self.input_dim = input_dim
        self.lsh_num_projs = lsh_num_projs
        self.block_size = block_size
        self.sample_size = sample_size
        self.min_seq_len = min_seq_len
        self.cuda = cuda
        self.lsh = AngularLSH(num_projs=self.lsh_num_projs, dim=(1, 1, input_dim))
        # self.lash = LSH(n_buckets=256, n_hashes=1, _rehash_each_round=True, dropout_rate=0.0, random_rotations_per_head=False)

        
    def forward(self, query: torch.tensor, key: torch.tensor, value: torch.tensor, scale=None, causal=False, return_lse=False):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        n_query = query.shape[2]
        batch_size, n_heads, n_key, dim = key.shape
        scale = dim ** (-0.5) if scale is None else scale
        
        # Without causal masking
        if not causal: 
            attn, lse = self.forward_no_causal_mask(query, key, value, scale)

        # With causal masking
        else:
            if n_key <= self.min_seq_len:
                if self.cuda:
                    attn, lse = exact_attention_cuda(query, key, value, scale, causal=True)
                else:
                    attn, lse = exact_attention(query, key, value, scale, causal=True)
            else:
            
                # If n_query is odd we pad inputs by adding all-zero rows
                if n_query % 2:
                    query = torch.nn.functional.pad(query, (0,0,0,1), mode='constant',value=0.)
                    key = torch.nn.functional.pad(key, (0,0,0,1), mode='constant',value=0.)
                    value = torch.nn.functional.pad(value, (0,0,0,1), mode='constant',value=0.)

                q_bd = query.view(batch_size, 2*n_heads, query.shape[2]//2, query.shape[-1])
                k_bd = key.view(batch_size, 2*n_heads, key.shape[2]//2, key.shape[-1])
                v_bd = value.view(batch_size, 2*n_heads, key.shape[2]//2, value.shape[-1])
        
                attn_bd, lse_bd = self.forward(q_bd, k_bd, v_bd, scale, True, True)
                
                if attn_bd.shape[2] not in attn_bd.stride():
                    attn_bd = attn_bd.contiguous()
                attn_bd = attn_bd.view(batch_size, n_heads, -1, dim)

                if lse_bd.shape[2] not in lse_bd.stride():
                    lse_bd = lse_bd.contiguous()
                lse_bd = lse_bd.view(batch_size, n_heads, -1, 1)

                attn_unmasked, lse_unmasked = self.forward_no_causal_mask(
                    query[:, :, key.shape[2]//2:, :],
                    key[:, :, :key.shape[2]//2, :], 
                    value[:, :, :key.shape[2]//2, :], scale)

                attn_up, lse_up = attn_bd[:,:,:query.shape[2]//2,:], lse_bd[:,:,:query.shape[2]//2,:]
                attn_down, lse_down = add_self_attentions(
                    attn_bd[:,:,query.shape[2]//2:,:],
                    lse_bd[:,:,query.shape[2]//2:,:],
                    attn_unmasked,
                    lse_unmasked)

                attn = torch.cat((attn_up, attn_down), dim=-2)
                lse = torch.cat((lse_up, lse_down), dim=-2)

                # If n_query was odd exclude the last rows
                if n_query % 2:
                    attn = attn[:,:,:-1,:]
                    lse = lse[:,:,:-1,:]

        if not return_lse:
            return attn
        else:
            return attn, lse


    def forward_no_causal_mask(self, query, key, value, scale):

        batch_size, head_size, n_query, dim = query.shape
        n_key = key.shape[2]

        if self.min_seq_len > n_query:
            if self.cuda:
                return exact_attention_cuda(query, key, value, scale, causal=False)
            else:
                return exact_attention(query, key, value, scale, causal=False)
        
        # 1. Sorted block-diagonal via sortLSH
        query_sort, query_sort_idx = torch.sort(self.lsh.hash(query), dim=2, stable=True) # batch_size x head_size x n
        _, key_sort_idx = torch.sort(self.lsh.hash(key), dim=2, stable=True)
        query_sort_idx_inv = torch.argsort(query_sort_idx, dim=2, stable=True) # for recovering the row order

        # get_distribution(query, query_sort, query_sort_idx_inv, self.lsh_num_projs, self.block_size, dim, self.lsh)

        key_block_size = self.block_size

        query_sorted = indexing(query, query_sort_idx, key_block_size)
        key_sorted = indexing(key, key_sort_idx, key_block_size)
        value_sorted = indexing(value, key_sort_idx, key_block_size)

        if key_block_size > 0:

            num_blocks = key_sorted.shape[2] // key_block_size
            query_block_size = query_sorted.shape[2] // num_blocks

            # Reshape tensors to [batch_size*head_size, 1, block_size, dim] as Flash-attn only allows 4d-tensors
            query_split_per_block = query_sorted.view(-1, 1, query_block_size, dim)
            key_split_per_block = key_sorted.view(-1, 1, key_block_size, dim)
            value_split_per_block = value_sorted.view(-1, 1, key_block_size, dim)

            if self.cuda:
                attn_block, lse_block = exact_attention_cuda(
                    query_split_per_block, key_split_per_block, value_split_per_block,
                    softmax_scale=scale, causal=False)
            else:
                attn_block, lse_block = exact_attention(
                    query_split_per_block, key_split_per_block, value_split_per_block,
                    softmax_scale=scale, causal=False)

            if attn_block.shape[2] not in attn_block.stride():
                attn_block = attn_block.contiguous()
            attn_block = attn_block.view(batch_size, head_size, query_sorted.shape[2], -1)

            if lse_block.shape[2] not in lse_block.stride():
                lse_block = lse_block.contiguous()
            lse_block = lse_block.view(batch_size, head_size, query_sorted.shape[2], -1)

            # When inputs are padded, then unpad them
            if query_sorted.shape[2] != n_query: #query.shape[2]:
                attn_block, lse_block = attn_block[:,:,:n_query,:], lse_block[:,:,:n_query,:]
                query_sorted = query_sorted[:,:,:n_query,:]
                key_sorted = key_sorted[:,:,:n_key,:]
                value_sorted = value_sorted[:,:,:n_key,:]

        else:
            query_block_size = -1
            query_block_size = -1
            attn_block, lse_block = 0, 0

        # 2. Residual low-rank part via uniform sampling
        # Sample indices uniformly at random
        sample_size = self.sample_size
        if sample_size > 0 and (n_query > query_block_size) and (n_key > key_block_size):
            sampled_set = torch.randint(n_key, size=(batch_size, head_size, sample_size), device=query_sorted.device)
            
            # Compute mask for hiding A_ij computed in block-diagonal attention
            offset_n = rearrange(torch.arange(n_query, device=query_sorted.device), 'n -> 1 n 1')
            weights = n_key / sample_size
            value_subset = indexing(value_sorted, sampled_set)
            key_subset = indexing(key_sorted, sampled_set)
            if not self.cuda:
                block_mask = (offset_n // query_block_size) == (sampled_set // key_block_size).view(-1, 1, sample_size)
                block_mask = block_mask.view(batch_size, head_size, -1, sample_size)
                block_mask = block_mask.to(query_sorted.dtype)
                block_mask *= torch.finfo(query_sorted.dtype).min # adding -inf added to QK^T

                attn_res, lse_res = exact_attention(query_sorted, key_subset, value_subset, scale, causal=False, bias=block_mask)
            else:
                attn_res, lse_res = exact_attention_cuda(query_sorted, key_subset, value_subset, scale, causal=False)

            lse_res = lse_res + math.log(weights)

            # Add two attentions
            if key_block_size > 0:
                attn, lse = add_self_attentions(attn_block, lse_block, attn_res, lse_res)
            else:
                attn, lse = attn_res, lse_res
        else:
            attn, lse = attn_block, lse_block

        # Re-order rows with the inverse order for query_sorted -> query
        attn = indexing(attn, query_sort_idx_inv)
        lse = indexing(lse, query_sort_idx_inv)
        return attn, lse

class ReformerAttention(torch.nn.Module):
    def __init__( self,
                  dropout = 0.,
                  bucket_size = 64,
                  n_hashes = 8,
                  causal = False,
                  allow_duplicate_attention = True,
                  attend_across_buckets = True,
                  rehash_each_round = True,
                  drop_for_hash_rate = 0.0,
                  random_rotations_per_head = False,
                  return_attn = False):
        super(ReformerAttention, self).__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = nn.Dropout(dropout)
        self.drop_for_hash_rate = drop_for_hash_rate
        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.bucket_size = bucket_size

        self.n_hashes = n_hashes

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

        # will expend extra computation to return attention matrix
        self._return_attn = return_attn

        # cache buckets for reversible network, reported by authors to make Reformer work at depth
        self._cache = {}

    def forward(self, qk, v, query_len = None, input_mask = None, input_attn_mask = None, pos_emb = None, **kwargs):
        batch_size, seqlen, dim = qk.shape
        device = qk.device
        
        self.n_buckets = seqlen // self.bucket_size
        lsh = LSH(self.n_buckets, self.n_hashes, _rehash_each_round = self._rehash_each_round, 
                  dropout_rate=self.drop_for_hash_rate, 
                  random_rotations_per_head=self._random_rotations_per_head, 
                  _allow_duplicate_attention=self._allow_duplicate_attention,
                  causal=self.causal)
        dots, undo_sort, bq_t, bkv_t, bv = lsh.hash_vectors(qk, v)

        # Softmax.
        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = torch.exp(dots - dots_logsumexp).type_as(dots)
        dropped_dots = self.dropout(dots)

        bo = torch.einsum('buij,buje->buie', dropped_dots, bv)
        so = torch.reshape(bo, (batch_size, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (batch_size, -1,))

        # unsort logits
        o = lsh.batched_index_select(so, undo_sort)
        logits = slogits.gather(1, undo_sort)

        o = torch.reshape(o, (batch_size, lsh.n_hashes, seqlen, dim))
        logits = torch.reshape(logits, (batch_size, lsh.n_hashes, seqlen, 1))

        if query_len != seqlen:
            query_slice = (slice(None), slice(None), slice(0, query_len))
            o, logits = o[query_slice], logits[query_slice]

        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        out = torch.sum(o * probs, dim=1)

        attn = torch.empty(0, device=device)

        # return unsorted attention weights
        if self._return_attn:
            attn_unsort = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            attn_unsort = attn_unsort.view(batch_size * lsh.n_hashes, -1).long()
            unsorted_dots = torch.zeros(batch_size * lsh.n_hashes, seqlen * seqlen, device=device)
            unsorted_dots.scatter_add_(1, attn_unsort, dots.view_as(attn_unsort))
            del attn_unsort
            unsorted_dots = unsorted_dots.reshape(batch_size, lsh.n_hashes, seqlen, seqlen)
            attn = torch.sum(unsorted_dots[:, :, 0:query_len, :] * probs, dim=1)

        return out, attn

class PrefillQAttention(nn.Module):
    def __init__(self, input_dim = 128, lsh_num_projs=7, block_size=32) -> None:
        super().__init__()
        self.lsh_num_projs = lsh_num_projs
        self.lsh = AngularLSH(num_projs=lsh_num_projs, dim=(1, 1, input_dim))
        self.block_size = block_size

    def visualize_lsh(self, q_lsh, k_lsh, layer_id, block_size):
        import matplotlib.pyplot as plt
        import numpy as np

        for i in range(block_size, q_lsh.shape[0], block_size):
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            ax[0].imshow(q_lsh[i-block_size:i].to(torch.float32).cpu().detach().numpy())
            ax[0].set_title("q lsh sorted")
            ax[1].imshow(k_lsh[i-block_size:i].to(torch.float32).cpu().detach().numpy())
            ax[1].set_title("k")
            plt.savefig(f"/home/lj9979/QUEST/tests/outputs/q_critical_layer_{layer_id}_{i/block_size}.png")
            plt.close(fig)

    def visualize_q_k(self, q, k, layer_id, seq_len):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import cm

        #plot a 3-d plot of q and k
        q = q.mean(dim=0)
        k = k.mean(dim=0)

        q = q.view(seq_len, -1)
        k = k.view(seq_len, -1)

        x = np.arange(q.shape[1])
        y = np.arange(q.shape[0])
        X, Y = np.meshgrid(x, y)

        q = q.cpu().to(torch.float32).detach().numpy()
        k = k.cpu().to(torch.float32).detach().numpy()

        q = abs(q)
        k = abs(k)

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_xlabel("Channels")
        ax.set_ylabel("Tokens")
        ax.plot_surface(X, Y, q, rstride=5, cstride=5, cmap=cm.coolwarm)
        ax.set_title("blocked_tensor")
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot_surface(X, Y, k, rstride=5, cstride=5, cmap=cm.coolwarm)
        ax.set_title("actual_tensor")
        ax.set_xlabel("Channels")
        ax.set_ylabel("Tokens")
        plt.savefig(f"/home/lj9979/QUEST/tests/outputs/q_k_layer_{layer_id}.png")
        plt.close(fig)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool,
        mask: Tensor,
    ) -> Tensor:
        self.n_head = q.shape[2]
        seq_len = q.shape[1]
        batch_size = q.shape[0]

        out_dir = xformers_attn(q, k, v, True, mask)
        
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        query_hash_sort, query_sort_idx = torch.sort(self.lsh.hash(q), dim=2, stable=True) # batch_size x head_size x n
        query_sort_idx_inv = torch.argsort(query_sort_idx, dim=2, stable=True) # for recovering the row order
        num_blocks = q.shape[2] // self.block_size
        query_block_size = self.block_size
        dim = q.shape[-1]

        query_sorted = indexing(q, query_sort_idx, query_block_size)
        query_sorted = query_sorted.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        out_actual_1 = xformers_attn(query_sorted, k, v, True, mask)
        out_actual_1 = out_actual_1.transpose(1,2)
        out_actual_1 = indexing(out_actual_1, query_sort_idx_inv, query_block_size)
        out_actual_1 = out_actual_1.transpose(1,2)
        query_sorted = query_sorted.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # print(out_actual_1.shape)
        # print(out_dir.shape)
        # query_unsort = indexing(query_sorted, query_sort_idx_inv, query_block_size)

        # assert torch.allclose(query_unsort, q, atol=1e-4)

        # assert torch.allclose(out_actual_1, out_dir, atol=1e-4)

        key_unsort = self.lsh.hash(k)
        key_unsort_unsort_hash_block = key_unsort.view(seq_len, -1)
        query_hash_sort_block = query_hash_sort.view(seq_len, -1)

        # Reshape tensors to [batch_size*head_size, 1, block_size, dim] as Flash-attn only allows 4d-tensors
        query_split_per_block = query_sorted.view(batch_size, num_blocks, -1, dim)
        K_split_per_block = k.reshape(batch_size, num_blocks, -1, dim)
        V_split_per_block = v.reshape(batch_size, num_blocks, -1, dim)

        blocks_moved = 0
        for i in range(0, num_blocks):
            count = torch.sum(torch.eq(query_hash_sort_block[i*query_block_size:(i+1)*query_block_size, :], key_unsort_unsort_hash_block[i*query_block_size:(i+1)*query_block_size, :])).item()
            if count == 0:
                K_split_per_block[:, i, :, :] = 0
            else:
                blocks_moved += 1
        
        print("blocks moved:", blocks_moved)
        q_critical = query_split_per_block.view(q.shape)
        k_critical = K_split_per_block.view(k.shape)
        v_critical = V_split_per_block.view(v.shape)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        q_critical = q_critical.transpose(1,2)
        k_critical = k_critical.transpose(1,2)
        v_critical = v_critical.transpose(1,2)

        out_actual = xformers_attn(q_critical, k, v, True, mask)
        out_actual = out_actual.transpose(1,2)
        out_actual = indexing(out_actual, query_sort_idx_inv, query_block_size)
        out_actual = out_actual.transpose(1,2)

        out_actual_no_sort = xformers_attn(q, k, v, True, mask)
        
        # self.visualize_q_k(q_critical, q, 0, seq_len)
        # self.visualize_q_k(k_critical, k, 1, seq_len)

        # assert torch.allclose(out_actual, out_actual_no_sort, atol=1e-4)

        out_critical = xformers_attn(q_critical, k_critical, v_critical, True, mask)
        out_critical = out_critical.transpose(1,2)
        out_critical = indexing(out_critical, query_sort_idx_inv, query_block_size)
        out_critical = out_critical.transpose(1,2)

        print("out_critical shape:", out_critical.shape)
        print("out_actual shape:", out_actual.shape)

        loss = ((out_critical - out_actual)**2).mean()
        print("loss after removing blocks:", loss.item())
        
        return out_critical
