import math
import torch
from einops import rearrange
import torch.nn as nn

from .utils import exact_attention, exact_attention_cuda, add_self_attentions, indexing, get_distribution
from .angular_lsh import AngularLSH, LSH

from torch import Tensor, nn

from .kv_cache import KVCache, DoubleKVCache, SingleKVCache
from dataclasses import dataclass
from typing import Literal

import torch.nn.functional as F

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

        get_distribution(query, query_sort, query_sort_idx_inv, self.lsh_num_projs, self.block_size, dim, self.lsh)

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
    
@dataclass(frozen=True)
class RKForCompressionRatio:
    """Config option to set r and k to achieve to specified compression ratio.

    i.e. ratio = 8 means SparQ will transfer ~1/8 of the data that would be transferred
    by dense.
    """

    ratio: int = 8
    
@dataclass(frozen=True)
class SparQArgs:
    rk: RKForCompressionRatio = RKForCompressionRatio(ratio=8)

    reallocation: bool = True
    running_V_mean: bool = True
    K_mode: Literal["store_once", "store_twice"] = "store_twice"
    # Sorting the output of the top-k takes time, but might result in more contiguous
    # memory accesses. In our experiments we found it was faster not to sort.
    sort_stage_1_top_k: bool = False
    sort_stage_2_top_k: bool = False

class RunningVMean(nn.Module):
    """Maintains a running mean of V over sequence length.

    FIXME: As the mean is accumulated for each token generated and reset of prefill,
    this implementation is only correct if the nth token is only generated once per
    prefill. The rest of gpt-fast doesn't enforce this, you can set input_pos to
    generate the nth token as many times as you like, but we ignore this problem for
    now.
    """

    def __init__(self, max_batch_size: int, n_local_heads: int, head_dim: int) -> None:
        super().__init__()
        self.register_buffer(
            "V_mean",
            torch.full(
                (max_batch_size, n_local_heads, 1, head_dim),
                float("nan"),
                dtype=torch.float32,
            ),
        )
        self.register_buffer("n", torch.tensor(0))

    def init(self, V: Tensor) -> None:
        self.V_mean[:, :, :, :] = V.mean(-2, dtype=torch.float32, keepdim=True)
        self.n.zero_().add_(V.shape[-2])

    def update(self, v: Tensor) -> Tensor:
        V_mean = (self.n * self.V_mean + v) / (self.n + 1)
        self.V_mean[:, :, :, :] = V_mean
        self.n.add_(1)
        return V_mean.to(v.dtype)


class PrefillQAttention(nn.Module):
    def __init__(self, n_local_heads, input_dim = 128, lsh_num_projs=7, block_size=32, sample_size=1024) -> None:
        super().__init__()
        self.n_local_heads = n_local_heads
        self.kv_cache: DoubleKVCache | SingleKVCache | None = None
        self.V_mean: RunningVMean | None = None
        self.r: int | None = None
        self.k: int | None = None
        self.K_mode = "store_twice"
        self.lsh_num_projs = lsh_num_projs
        self.lsh = AngularLSH(num_projs=lsh_num_projs, dim=(1, 1, input_dim))
        self.block_size = block_size
        self.rk_ratio = 2

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor,
        input_pos: Tensor | None,
        prefill: bool,
    ) -> Tensor:
        self.n_head = q.shape[2]
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        if self.kv_cache is not None:
            K1, K2, V = self.kv_cache.update(input_pos, k, v)

        K1 = K1.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        K2 = K2.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        V = V.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        #    q: [batch x n heads         x target seq len x head embed dim]
        #    K: [batch x n grouped heads x source seq len x head embed dim]
        #    V: [batch x n grouped heads x source seq len x head embed dim]
        # mask: [1     x 1               x 1              x source seq len]

        if prefill:
            return self._prefill(q, k, v, mask)
        else:
            return self._generate(q, K1, K2, V, v, mask)

    def _prefill(self, q: Tensor, K: Tensor, V: Tensor, mask: Tensor) -> Tensor:
        # self.V_mean.init(V)
        query_hash_sort, query_sort_idx = torch.sort(self.lsh.hash(q), dim=2, stable=True) # batch_size x head_size x n
        num_blocks = q.shape[2] // self.block_size
        query_block_size = q.shape[2] // num_blocks
        dim = q.shape[-1]

        query_sorted = indexing(q, query_sort_idx, query_block_size)

        key_unsort = self.lsh.hash(K)
        key_unsort_unsort_hash_block = key_unsort.view(-1, 1, query_block_size)
        query_hash_sort_block = query_hash_sort.view(-1, 1, query_block_size)

        # Reshape tensors to [batch_size*head_size, 1, block_size, dim] as Flash-attn only allows 4d-tensors
        query_split_per_block = query_sorted.view(-1, 1, query_block_size, dim)
        K_split_per_block = K.reshape(-1, 1, query_block_size, dim)
        V_split_per_block = V.reshape(-1, 1, query_block_size, dim)

        # calculate criticality score
        criticality = torch.sum(key_unsort_unsort_hash_block == query_hash_sort_block, dim=-1).squeeze()
        critical_indices = torch.where(criticality > 0)
        critical_mask = torch.ones_like(K_split_per_block)
        critical_indices = critical_indices[0].cpu().tolist()
        # torch.index_select(critical_mask, 0, critical_indices)
        for x in critical_indices:
            critical_mask[x, :, :, :] = 0

        print("Number of blocks moved:", K_split_per_block.shape[0]-len(critical_indices))

        K_split_per_block = K_split_per_block * critical_mask

        out_critical = F.scaled_dot_product_attention(query_split_per_block, K_split_per_block, V_split_per_block, mask)
        out_actual = F.scaled_dot_product_attention(q, K, V, mask)

        out_critical = out_critical.view(out_actual.shape)

        loss = ((out_critical - out_actual)**2).mean()
        print("loss after removing blocks:", loss.item())
        
        return F.scaled_dot_product_attention(q, K, V, mask)

    def _generate(
        self, q: Tensor, K1: Tensor, K2: Tensor, V: Tensor, v: Tensor, mask: Tensor, config: SparQArgs
    ) -> Tensor:
        if config.reallocation and config.running_V_mean:
            V_mean = self.V_mean.update(v)
        elif config.reallocation and not config.running_V_mean:
            V_mean = self._masked_V_mean(V, mask)
        else:
            V_mean = None
        assert self.r is not None and self.k is not None
        return self.sparq_attn(q, K1, K2, V, V_mean, mask, self.r, self.k, self.config)

    def _masked_V_mean(self, V: Tensor, mask: Tensor) -> Tensor:
        value_mask = mask.transpose(-2, -1)
        V_sum = (V * value_mask).sum(-2, dtype=torch.float32, keepdim=True)
        V_mean = V_sum / value_mask.sum(-2, dtype=torch.float32, keepdim=True)
        return V_mean.to(V.dtype)
    
    def get_r_k_for_compression_ratio(
        self,
        sequence_length: int, head_dim: int
    ) -> tuple[int, int]:
        """Gets r, k to reduce memory transferred during attention by the given ratio."""
        r = round(head_dim / self.rk_ratio)
        k = round(sequence_length / (2 * self.rk_ratio))
        return r, k
    
    def setup_caches(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype=torch.bfloat16,
    ) -> None:
        if self.K_mode == "store_twice":
            self.kv_cache = DoubleKVCache(
                max_batch_size, max_seq_length, n_heads, head_dim, dtype
            )
        elif self.K_mode == "store_once":
            self.kv_cache = SingleKVCache(
                max_batch_size, max_seq_length, n_heads, head_dim, dtype
            )

        self.V_mean = RunningVMean(max_batch_size, n_heads, head_dim)
        self.r, self.k = self.get_r_k_for_compression_ratio(
            max_seq_length, head_dim
        )

    @torch.compile(disable=not torch.cuda.is_available())
    def _scaled_softmax(self, x: Tensor, divscale: Tensor | float, dim: int) -> Tensor:
        return torch.softmax(x / divscale, dim=dim)

    def sparq_attn(
        self,
        Q: Tensor,
        K1: Tensor,
        K2: Tensor,
        V: Tensor,
        V_mean: Tensor | None,
        mask: Tensor,
        r: int,
        k: int,
        config: SparQArgs,
    ) -> Tensor:
        # 1. Approximate attention scores using r largest components of Q
        absQ = torch.abs(Q)
        absQ_hat, i1 = torch.topk(absQ, r, dim=-1, sorted=config.sort_stage_1_top_k)
        QK_hat = _gather(Q, -1, i1) @ _gather(K1, -1, i1).transpose(-1, -2)
        masked_QK_hat = torch.where(mask, QK_hat, float("-inf"))
        scale = torch.sqrt(
            Q.shape[-1]
            * absQ_hat.sum(dim=-1, keepdim=True)
            / absQ.sum(dim=-1, keepdim=True)
        )
        s_hat = self._scaled_softmax(masked_QK_hat, scale, dim=-1)

        # 2. Gather top k2 positions based on approximate attention scores & run attention
        # This min ensures that k <= sequence length, otherwise torch.compile() will crash.
        k = min(k, V.shape[-2])
        s_hat_i2, i2 = torch.topk(s_hat, k, dim=-1, sorted=config.sort_stage_2_top_k)
        iKV = i2[..., 0, :, None]
        QK = Q @ _gather(K2, -2, iKV).transpose(2, 3)
        masked_QK = torch.where(_gather(mask.expand_as(QK_hat), -1, i2), QK, float("-inf"))
        s = self._scaled_softmax(masked_QK, Q.shape[-1] ** 0.5, dim=-1)
        y_ = s @ _gather(V, -2, iKV)

        # 3. Estimate the total score of the top k, and interpolate with V_mean
        if V_mean is not None:
            return torch.lerp(V_mean, y_, s_hat_i2.sum(-1, keepdim=True))
        else:
            return y_


def _gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
    dim += (dim < 0) * t.ndim
    return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))

