NUM_TOTAL_LAYERS = {
    'chatglm2-6b-32k': 28,
    'opt-350m-50k': 12,
    'glm-edge-1.5b-chat': 28,
    'open_lm_1b': 24,
}

def patch_attention_layers(model_config, model_name, patch_config, num_patch_layers, **kwargs):

    num_total_layers = NUM_TOTAL_LAYERS[model_name]
    num_patch_layers = num_total_layers if num_patch_layers < 0 else num_patch_layers
    
    if patch_config == 'last':
        patch_layer_indices = range(num_total_layers-1, num_total_layers-num_patch_layers-1, -1)

    elif patch_config == 'first':
        patch_layer_indices = range(num_patch_layers)
        
    elif patch_config == 'odd':
        patch_layer_indices = range(1, num_total_layers, 2)

    elif patch_config == 'even':
        patch_layer_indices = range(0, num_total_layers, 2)

    elif patch_config == 'odd_first':
        patch_layer_indices = range(1, 2*num_patch_layers, 2)

    elif patch_config == 'odd_last':
        patch_layer_indices = range(num_total_layers-1, num_total_layers-num_patch_layers, -1)

    elif patch_config == 'even_first':
        patch_layer_indices = range(0, num_total_layers, 2)[:num_patch_layers]

    elif patch_config == 'even_last':
        patch_layer_indices = range(1, num_total_layers, 2)[-num_patch_layers:]

    else:
        raise NotImplementedError(f"Invalid patch_config option: {patch_config}")

    if model_name == 'glm-edge-1.5b-chat':
        from .modeling_fast_attention import FastCoreAttention
    
        print(f"patch_config: {patch_config}, attn_method: {kwargs['attn_method']}, num_patch_layers: {num_patch_layers}, patch_indices: {list(patch_layer_indices)}")
        for i in patch_layer_indices:
            model_config.model.layers[i].self_attn.core_attention = FastCoreAttention(model_config.config, i, **kwargs)
    
    elif model_name == 'chatglm2-6b-32k':
        from .modeling_fast_attention import FastCoreAttention
    
        print(f"patch_config: {patch_config}, attn_method: {kwargs['attn_method']}, num_patch_layers: {num_patch_layers}, patch_indices: {list(patch_layer_indices)}")
        for i in patch_layer_indices:
            model_config.transformer.encoder.layers[i].self_attention.core_attention = FastCoreAttention(model_config.config, i, **kwargs)
    
    elif model_name == 'opt-350m-32k':
        from .modeling_fast_attention import FastCoreAttention
    
        print(f"patch_config: {patch_config}, attn_method: {kwargs['attn_method']}, num_patch_layers: {num_patch_layers}, patch_indices: {list(patch_layer_indices)}")
        for i in patch_layer_indices:
            model_config.model.decoder.layers[i].self_attn.core_attention = FastCoreAttention(model_config.config, i, **kwargs)

    elif model_name == 'open_lm_1b':
        from .modeling_fast_attention import CustomOpenLmAttn
    
        print(f"patch_config: {patch_config}, attn_method: {kwargs['attn_method']}, num_patch_layers: {num_patch_layers}, patch_indices: {list(patch_layer_indices)}")
        for i in patch_layer_indices:
            model_config.layers[i].attention = CustomOpenLmAttn(kwargs['param_config'], i, **kwargs)
