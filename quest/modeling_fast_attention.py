import math
import torch
from typing import Callable, Optional, Tuple

import torch.nn as nn

from .quest_attn import HyperAttention, PrefillQAttention
from open_lm.attention import xformers_attn, torch_attn

from open_lm.positional_embedding.head_rotary import HeadRotaryWithCast
from open_lm.positional_embedding.rotary import RotaryWithCast
from open_lm.positional_embedding.llama_rotary import LLaMARotaryWithCast
from open_lm.positional_embedding.none import identity_with_cast

# Edited from https://huggingface.co/THUDM/chatglm2-6b-32k/blob/main/modeling_chatglm.py#L194
class FastCoreAttention(torch.nn.Module):
    
    def __init__(self, config, layer_number, **kwargs):
        super(FastCoreAttention, self).__init__()

        if hasattr(config, 'apply_query_key_layer_scaling'):
            self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        else:
            self.apply_query_key_layer_scaling = False
        if hasattr(config, 'attention_softmax_in_fp32'):
            self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        else:
            self.attention_softmax_in_fp32 = False
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        if hasattr(config, 'kv_channels'):
            projection_size = config.kv_channels * config.num_attention_heads
        if hasattr(config, 'num_key_value_heads'):
            projection_size = config.num_key_value_heads * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.num_attention_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)
        
        self.attn_method = kwargs.get('attn_method')
        if self.attn_method in ['hyper', 'hyper-cuda']:
            lsh_num_projs = kwargs.get('lsh_num_projs')
            block_size = kwargs.get('block_size')
            sample_size = kwargs.get('sample_size')
            min_seq_len = kwargs.get('min_seq_len')
            self.attn = HyperAttention(
                input_dim=128,
                lsh_num_projs=lsh_num_projs, 
                block_size=block_size,
                sample_size=sample_size, 
                min_seq_len=min_seq_len,
                cuda='cuda' in self.attn_method)
        elif self.attn_method == 'prefill':
            lsh_num_projs = kwargs.get('lsh_num_projs')
            block_size = kwargs.get('block_size')
            min_seq_len = kwargs.get('min_seq_len')
            self.attn = PrefillQAttention(
                n_local_heads=1,
                lsh_num_projs=lsh_num_projs, 
                block_size=block_size)
        else: 
            raise NotImplementedError("Invalid attn_method option")
        

    def forward(self, query_layer, key_layer, value_layer, attention_mask):

        query_layer, key_layer, value_layer = [k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]
        if self.attn_method in ['hyper', 'hyper-cuda'] and attention_mask is None and query_layer.shape[2] == key_layer.shape[2]:
            softmax_scale = query_layer.shape[-1]**(-0.5)
            context_layer = self.attn(query_layer, key_layer, value_layer, causal=False)
        elif self.attn_method == 'prefill':
            context_layer = self.attn(query_layer, key_layer, value_layer, attention_mask)
        else:
            assert False, 'this part the query length and key length may be different and not be a computational bottleneck.'
            if attention_mask is not None:
                attention_mask = ~attention_mask
            context_layer = torch.nn.functional.scaled_dot_product_attention(query_layer, key_layer, value_layer, attention_mask)

        context_layer = context_layer.permute(2, 0, 1, 3)
        print(context_layer.shape)
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.reshape(*new_context_layer_shape)
        return context_layer
    

# Edited from https://github.com/mlfoundations/open_lm/blob/main/open_lm/model.py
class Params:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1
    norm_eps: float = 1e-5
    seq_len: int = 2048
    post_embed_norm: bool = False
    weight_tying: bool = False
    norm_type: torch.nn.Module = torch.nn.LayerNorm
    attn_func: Callable = xformers_attn if torch.cuda.is_available() else torch_attn
    apply_qk_norm: bool = True
    moe_loss_weight: float = 0.1
    moe_capacity_factor: float = 1.25
    moe_expert_model_parallelism: bool = False
    moe_weight_parallelism: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_freq: int = 0
    positional_embedding_type: str = "rotary"
    ffn_type: str = "swiglu"

# Edited from https://github.com/mlfoundations/open_lm/blob/main/open_lm/model.py#L117

class CustomOpenLmAttn(torch.nn.Module):
    def __init__(self, args, layer_id, **kwargs):
        super().__init__()

        self.layer_id = layer_id
        self.attn_method = kwargs.get('attn_method')
        if self.attn_method in ['hyper', 'hyper-cuda']:
            lsh_num_projs = kwargs.get('lsh_num_projs')
            block_size = kwargs.get('block_size')
            sample_size = kwargs.get('sample_size')
            min_seq_len = kwargs.get('min_seq_len')
            self.attn_func = HyperAttention(
                input_dim=128,
                lsh_num_projs=lsh_num_projs, 
                block_size=block_size,
                sample_size=sample_size, 
                min_seq_len=min_seq_len,
                cuda='cuda' in self.attn_method) 
        elif self.attn_method == 'prefill':
            lsh_num_projs = kwargs.get('lsh_num_projs')
            block_size = kwargs.get('block_size')
            self.attn_func = PrefillQAttention(
                lsh_num_projs=lsh_num_projs, 
                block_size=block_size)
        else: 
            raise NotImplementedError("Invalid attn_method option")

    def forward(self, queries, keys, vals, is_causal=True, past_key_value=None, use_cache=False, attention_mask=None):
        if past_key_value is not None and use_cache:
            keys = torch.cat([past_key_value[0], keys], dim=1)
            vals = torch.cat([past_key_value[1], vals], dim=1)

        if use_cache:
            past_key_value = [keys, vals]

        if self.attn_method in ['hyper', 'hyper-cuda'] and attention_mask is None and queries.shape[2] == keys.shape[2]:
            softmax_scale = queries.shape[-1]**(-0.5)
            output = self.attn_func(queries, keys, vals, causal=False)
        elif self.attn_method == 'prefill':
            output = self.attn_func(queries, keys, vals, is_causal, attention_mask)
        else:
            assert False, 'this part the query length and key length may be different and not be a computational bottleneck.'
        return output
    
