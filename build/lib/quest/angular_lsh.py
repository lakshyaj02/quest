import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F

TOKEN_SELF_ATTN_VALUE = -5e4 # carefully set for half precision to work

class AngularLSH(torch.nn.Module):

    def __init__(self, num_projs, dim, rng=None):
        super().__init__()
        self.num_projs = num_projs

        if num_projs > 0:
            self.register_buffer('proj_dir', torch.randn(dim + (num_projs,), generator=rng), persistent=False) # [1, 1, dim, num_projs] just a random matrix
            self.register_buffer('perm', self._unit_hamming_distance_array(self.num_projs), persistent=False) # Array of numbers from 0 to 2^num_projs, arranged according to their hamming distances
            self.register_buffer('enc_vec', 2 ** torch.arange(self.num_projs).view(1, 1, 1, -1), persistent=False) # [1, 1, 1, num_projs], geometric series from 2^0 to 2^(num_projs-1)
            
    def _unit_hamming_distance_array(self, size_n):
        if size_n == 1:
            return torch.tensor([0, 1])
        a = self._unit_hamming_distance_array(size_n - 1)
        return torch.concat([a, torch.flip(a, dims=[0]) + 2 ** (size_n - 1)], 0)

    def hash(self, mat):
        if self.num_projs < 0:
            return torch.zeros(mat.shape[:-1], device=mat.device, dtype=torch.int32)
        mask = torch.einsum('...nd,...dr -> ...nr', mat, self.proj_dir.to(mat.device))
        mask = mask > 0
        bin_ids = (mask * self.enc_vec.to(mask.device)).sum(-1)
        return self.perm[bin_ids]
    
    def __repr__(self):
        return f"AngularLSH(num_proj={self.num_projs}, proj_dir.shape={self.proj_dir.shape})"
    

class LSH(torch.nn.Module):
    def __init__(self, 
                 n_buckets, 
                 n_hashes, 
                 _rehash_each_round=True, 
                 dropout_rate = 0.0, 
                 random_rotations_per_head = False, 
                 _allow_duplicate_attention = True,
                 causal = False):
        super(LSH, self).__init__()
        self.n_buckets = n_buckets
        self.n_hashes = n_hashes
        self._rehash_each_round = _rehash_each_round
        self.dropout_for_hash = nn.Dropout(dropout_rate)
        self._random_rotations_per_head = random_rotations_per_head
        self._attend_across_buckets = True
        self._allow_duplicate_attention = _allow_duplicate_attention
        self.causal = causal

    def sort_key_val(self, t1, t2, dim=-1):
        values, indices = t1.sort(dim=dim)
        t2 = t2.expand_as(t1)
        return values, t2.gather(dim, indices)
    
    def batched_index_select(self, values, indices):
        last_dim = values.shape[-1]
        return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))
    
    def rotate_every_two(self, x):
        x = rearrange(x, '... (d j) -> ... d j', j = 2)
        x1, x2 = x.unbind(dim = -1)
        x = torch.stack((-x2, x1), dim = -1)
        return rearrange(x, '... d j -> ... (d j)')
    
    def apply_rotary_pos_emb(self, qk, sinu_pos):
        sinu_pos = sinu_pos.type(qk.dtype)
        sinu_pos = rearrange(sinu_pos, '() n (j d) -> n j d', j = 2)
        sin, cos = sinu_pos.unbind(dim = -2)
        sin, cos = map(lambda t: repeat(t, 'n d -> n (d j)', j = 2), (sin, cos))
        seq_len = sin.shape[0]
        qk, qk_pass = qk[:, :seq_len], qk[:, seq_len:]
        qk = (qk * cos) + (self.rotate_every_two(qk) * sin)
        return torch.cat((qk, qk_pass), dim = 1)
    
    def max_neg_value(self, tensor):
        return -torch.finfo(tensor.dtype).max
    
    def mask_for_padding(self, dots, masked_value, st, seqlen, batch_size, chunk_size, input_mask = None):
        if input_mask is not None:
            input_mask = F.pad(input_mask, (0, seqlen - input_mask.shape[1]), value=True)
            mq = input_mask.gather(1, st).reshape((batch_size, chunk_size, -1))
            mkv = self.look_one_back(mq)
            mask = mq[:, :, :, None] * mkv[:, :, None, :]
            dots.masked_fill_(~mask, masked_value)
            del mask

        return dots
    
    def mask_post_attention(self, dots, masked_value, bq_t, bkv_t, seqlen, batch_size, input_attn_mask = None):
        if input_attn_mask is not None:
            input_attn_mask = F.pad(input_attn_mask, (0, seqlen - input_attn_mask.shape[-1], 0, seqlen - input_attn_mask.shape[-2]), value=True)
            dot_attn_indices = ((bq_t * seqlen)[:, :, :, None] + bkv_t[:, :, None, :])
            input_attn_mask = input_attn_mask.reshape(batch_size, -1)
            dot_attn_indices = dot_attn_indices.reshape(batch_size, -1)
            mask = input_attn_mask.gather(1, dot_attn_indices).reshape_as(dots)
            dots.masked_fill_(~mask, masked_value)
            del mask

        return dots
    
    def chunked_sum(self, tensor, chunks=1):
        *orig_size, last_dim = tensor.shape
        tensor = tensor.reshape(-1, last_dim)
        summed_tensors = [c.sum(dim=-1) for c in tensor.chunk(chunks, dim=0)]
        return torch.cat(summed_tensors, dim=0).reshape(orig_size)

    def hash_vectors(self, vecs, v, input_mask=None, input_attn_mask = None, pos_emb = None):
        batch_size, seqlen, dim = vecs.shape
        device = vecs.device

        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert self.n_buckets % 2 == 0

        rot_size = self.n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            vecs.shape[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)

        random_rotations = torch.randn(rotations_shape, dtype=vecs.dtype, device=device).expand(batch_size, -1, -1, -1)

        dropped_vecs = self.dropout_for_hash(vecs)
        rotated_vecs = torch.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)

        if self._rehash_each_round:
            # rotated_vectors size [batch,n_hash,seq_len,buckets]
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            buckets = torch.argmax(rotated_vecs, dim=-1)
        else:
            rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = torch.squeeze(rotated_vecs, 1)
            bucket_range = torch.arange(rotated_vecs.shape[-1], device=device)
            bucket_range = torch.reshape(bucket_range, (1, -1))
            bucket_range = bucket_range.expand_as(rotated_vecs)

            _, buckets = self.sort_key_val(rotated_vecs, bucket_range, dim=-1)
            # buckets size [batch size, seq_len, buckets]
            buckets = buckets[... , -self.n_hashes:].transpose(1, 2)

        # buckets is now (self.n_hashes, seq_len). Next we add offsets so that
        # bucket numbers from different hashing rounds don't overlap.
        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * self.n_buckets, (1, -1, 1))
        buckets = torch.reshape(buckets + offsets, (batch_size, -1,))

        assert int(buckets.shape[1]) == self.n_hashes * seqlen
        total_hashes = self.n_hashes

        ticker = torch.arange(total_hashes * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        buckets_and_t = seqlen * buckets + (ticker % seqlen)
        buckets_and_t = buckets_and_t.detach()

        # Perform hash based sort ("s" at the beginning of the variable number is to indicate that it has been sorted)
        sbuckets_and_t, sticker = self.sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sticker.sort(dim=-1)
        del ticker

        sbuckets_and_t = sbuckets_and_t.detach()
        undo_sort = undo_sort.detach()
        sticker = sticker.detach()

        if pos_emb is not None:
            vecs = self.apply_rotary_pos_emb(vecs, pos_emb)

        st = (sticker % seqlen)
        sqk = self.batched_index_select(vecs, st)
        sv = self.batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        chunk_size = total_hashes * self.n_buckets
        bq_t = bkv_t = torch.reshape(st, (batch_size, chunk_size, -1))
        bqk = torch.reshape(sqk, (batch_size, chunk_size, -1, dim))
        bv = torch.reshape(sv, (batch_size, chunk_size, -1, dim))
        
        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)

        # Dot-product attention.
        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)
        masked_value = self.max_neg_value(dots)

        dots = self.mask_post_attention(dots, masked_value, bq_t, bkv_t, seqlen, batch_size, input_attn_mask)

        dots = self.mask_for_padding(dots, masked_value, st, seqlen, batch_size, chunk_size, input_mask)

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
            if seqlen > seqlen:
                mask = mask & (bkv_t[:, :, None, :] < seqlen)
            dots.masked_fill_(mask, masked_value)
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots.masked_fill_(self_mask, TOKEN_SELF_ATTN_VALUE)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bq_buckets = bkv_buckets = torch.reshape(sbuckets_and_t // seqlen, (batch_size, chunk_size, -1))
            bkv_buckets = look_one_back(bkv_buckets)
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots.masked_fill_(bucket_mask, masked_value)
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % chunk_size
            if not self._attend_across_buckets:
                locs1 = buckets * chunk_size + locs1
                locs2 = buckets * chunk_size + locs2
            locs = torch.cat([
                torch.reshape(locs1, (batch_size, total_hashes, seqlen)),
                torch.reshape(locs2, (batch_size, total_hashes, seqlen)),
            ], 1).permute((0, 2, 1))

            slocs = self.batched_index_select(locs, st)
            b_locs = torch.reshape(slocs, (batch_size, chunk_size, -1, 2 * total_hashes))

            b_locs1 = b_locs[:, :, :, None, :total_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, total_hashes))
            bq_locs = torch.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = self.chunked_sum(dup_counts, chunks=(total_hashes * batch_size))
            dup_counts = dup_counts.detach()
            assert dup_counts.shape == dots.shape
            dots = dots - torch.log(dup_counts + 1e-9)
            del dup_counts

        return dots, undo_sort, bq_t, bkv_t, bv