import torch
from torch import Tensor, nn

# Copyright (c) Graphcore 2024
# All rights reserved.
# This source code is licensed under the BSD-3 license,
# see the LICENSE file in the root directory of this source tree.


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val) -> tuple[Tensor, Tensor]:
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out
    
class DoubleKVCache(nn.Module):
    """KV cache that stores both K and K transpose."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype,
    ) -> None:
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer(
            "kt_cache",
            torch.zeros(cache_shape, dtype=dtype).transpose(-1, -2).contiguous(),
        )
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(
        self, input_pos: Tensor, k_val: Tensor, v_val: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out: Tensor = self.k_cache
        kt_out: Tensor = self.kt_cache
        v_out: Tensor = self.v_cache
        k_out[:, :, input_pos] = k_val.to(k_out.device)
        kt_out[:, :, :, input_pos] = k_val.transpose(-1, -2).to(kt_out.device)
        v_out[:, :, input_pos] = v_val.to(v_out.device)

        return kt_out.transpose(-1, -2), k_out, v_out


class SingleKVCache(nn.Module):
    """Just wraps the default KV cache so it has the same interface as DoubleKVCache."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_length: int,
        n_heads: int,
        head_dim: int,
        dtype,
    ) -> None:
        super().__init__()
        self.cache = KVCache(max_batch_size, max_seq_length, n_heads, head_dim, dtype)

    def update(
        self, input_pos: Tensor, k_val: Tensor, v_val: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        K, V = self.cache.update(input_pos, k_val, v_val)
        return K, K, V
