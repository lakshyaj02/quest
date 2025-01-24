import math
import torch
import matplotlib.pyplot as plt
import numpy as np
try:
    from flash_attn import flash_attn_func as flash_attn_func_cuda
except ImportError:
    flash_attn_func_cuda = None

from .flash_attn_triton import flash_attn_func

from .angular_lsh import AngularLSH, LSH

def find_indices(tensor, value):
    return (tensor == value).nonzero().squeeze(0)

def heatmaps(data1, data2, method='pearson'):
    # print(data1.shape, data1)
    plt.figure()
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))

    ax1, ax2 = axes

    im1 = ax1.matshow(data1)
    im2 = ax2.matshow(data2)

    ax1.set_title("Hashed Values")
    ax2.set_title("L2 Norms")
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    plt.tight_layout()
    plt.savefig("heatmap.png")
    plt.close()

def print_distribution(tensor, norm, num_projs):
    tensor = tensor.squeeze(0).cpu().numpy()
    norms = torch.norm(norm, dim=-1, p=2).squeeze(0).cpu().float().numpy()
    
    # plt.hist(tensor, bins=2**num_projs, range=(0, 2**num_projs), align='left', rwidth=0.8)
    # plt.xlabel("Bucket Value")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of Hashed Values")
    # plt.savefig("distribution.png")
    heatmaps(tensor, norms)

def get_cmap(n, name='rainbow'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def get_distribution(x, x_sort, inv_idx, num_projs, block_size, dim, mini_lsh = None):
    num_blocks = x.shape[2] // block_size
    query_block_size = x.shape[2] // num_blocks

    x_block = x.view(-1, 1, query_block_size, dim) 
    x_block_hash = torch.permute(x_block, [1, 2, 0, 3])

    if mini_lsh is None:
        mini_lsh = AngularLSH(num_projs=num_projs, dim=(1, 1, dim))
    
    # query_sort, query_sort_idx = torch.sort(mini_lsh.hash(x_block_hash), dim=2, stable=True) # batch_size x head_size x n
    query_unsort = mini_lsh.hash(x_block_hash)
    # x_unsort = mini_lsh.hash(x)
    # x_unsort_block = x_unsort.view(-1, 1, query_block_size, dim)
    # query_idx = torch.arange(0, query_unsort.shape[2], device=query_unsort.device).unsqueeze(0).unsqueeze(0).expand(query_unsort.shape[0], query_unsort.shape[1], -1)

    x_norm = x.view(-1, dim)
    x_norm_reshape = x_norm.view(-1, query_unsort.shape[1], query_unsort.shape[2], dim)
    norms = torch.norm(x_norm, dim=1, p=2)
    x_lab = np.arange(0, x_norm.shape[0])

    # Plot the norms
    plt.figure()
    plt.plot(x_lab, norms.cpu().float().numpy(), "*")
    plt.xlabel("Vector Index")
    plt.ylabel("L2 Norm")
    plt.title("L2 Norm of Vectors in a Tensor")
    plt.savefig("norms.png")
    plt.close()

    print(block_size, num_blocks)
    for blk_i in range(0, num_blocks):
        print_distribution(query_unsort[:,:,blk_i*block_size:(blk_i+1)*block_size], x_norm_reshape[:,:,blk_i*block_size:(blk_i+1)*block_size,:], num_projs)

    dist_list = []
    for blk_i in range(0, num_blocks):
        block_list = []
        for i in range(0, 2**num_projs):
            ind_i = find_indices(query_unsort[:,:,blk_i*block_size:(blk_i+1)*block_size], i)
            block_list.append(ind_i.shape[0])
        dist_list.append(block_list)

    # x_block_hashed = x_block_hash.gather(2, query_idx.unsqueeze(-1).expand(-1, -1, -1, x_block_hash.shape[-1])) # torch.Size([1, 256, 4096, 128])

    bins_list = [i for i in range(0, 2**num_projs)]
    
    # plt.bar(x=bins_list, height=dist_list, width=0.8, align='center', color='blue', edgecolor='black')
    w = 1/(2*num_blocks)
    width_range = torch.arange(-num_blocks*w, num_blocks*w, 2*w)
    width_range = width_range.tolist()

    cmap = get_cmap(len(width_range))

    plt.figure(figsize=(10, 6))

    for i in range(len(width_range)):
    # for i in range(1):
        plt.bar([j + width_range[i] for j in bins_list], dist_list[i], width=w*100, align='center', color=cmap(i))
    
    # Customize the plot
    plt.title('Sample Histogram')
    plt.xlabel('Bucket Value')
    plt.ylabel('Frequency')
    plt.savefig('sample_histogram.png')
    plt.close()

def indexing(x, indices, chunk_size=-1, dist = True):
    """ 
    inputs:
        - x: 4d-tensor with shape [b, h, n, d] 
        - indices: 3d-tensor with shape [b, h, s] where each entry should be in [0, n-1]
    output:
        - out: 4d-tensor with shape [b, h, s, d] where out[i,j] = x[i,j][indices[i,j],:]
    
    A naive implementation:
        out = torch.zeros(b, h, s, d)
        for i in range(b):
            for j in range(h):
                out[i,j] = x[i,j][idx[i,j],:]
        return out
    """
    if chunk_size < 0 or (chunk_size > 0 and x.shape[-2] % chunk_size == 0):
        return x.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
    else:
        x = x.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
        new_n = math.ceil(x.shape[2] / chunk_size) * chunk_size
        if new_n <= 0 or new_n - x.shape[2] <= 0:
            import pdb; pdb.set_trace();
        return torch.nn.functional.pad(x, (0,0,0,new_n-x.shape[2]), mode='constant',value=0.)


def add_self_attentions(attn1, lse1, attn2, lse2):
    """
    inputs:
        - attn1, attn2: 4d-tensors with shape [b, h, n, d]
        - lse1, lse2: 4d-tensors of log-sum-exp with shape [b, h, n, 1]
    output:
        - attn
        = (attn1 * exp(lse1) + attn2 * exp(lse2)) / (exp(lse1) + exp(lse2))
        = (attn1 + attn2 * exp(lse2 - lse1)) / (1 + exp(lse2-lse1))
        = attn1 * c + attn2 * (1-c), where c=1/(1 + exp(lse2-lse1)),
        - lse 
        = log(exp(lse1) + exp(lse2)) 
        = log(exp(lse1) * (1 + exp(lse2 - lse1))) 
        = lse1 + log(1 + exp(lse2 - lse1)) = lse1 - log(c)
    """
    c = (1 / (1 + (lse2 - lse1).exp())).to(dtype=attn1.dtype)
    attn = c * attn1 + (1-c) * attn2
    lse = lse1 - (c + torch.finfo(lse1.dtype).eps).log()
    return attn, lse


def exact_attention(query, key, value, softmax_scale, causal=False, bias=None):
    if query.dtype not in [torch.bfloat16, torch.float16]:
        qk = query @ key.transpose(-1,-2) * softmax_scale
        if causal:
            qk += (torch.ones(query.shape[2], key.shape[2], device=query.device) * torch.finfo(query.dtype).min).triu(1).reshape(1,1,query.shape[2], key.shape[2])
        out = qk.softmax(dim=-1) @ value
        lse = torch.logsumexp(qk, dim=-1, keepdim=True)
        return out, lse

    out, lse = flash_attn_func(
        query.transpose(1,2), key.transpose(1,2), value.transpose(1,2),
        bias, causal, softmax_scale)
    out = out.transpose(1,2)
    
    lse = lse.detach()
    if lse.shape[2] != out.shape[2]:
        lse = lse[:,:,:out.shape[2]]
    lse = lse.unsqueeze(-1)
    return out, lse
    

def exact_attention_cuda(query, key, value, softmax_scale, causal, bias=None):
    if flash_attn_func_cuda is None:
        raise ImportError("Please install flash_attn (pip install flash-attn --no-build-isolation)")
    out, lse, _ = flash_attn_func_cuda(
        query.transpose(1,2), key.transpose(1,2), value.transpose(1,2),
        softmax_scale=softmax_scale, causal=causal, return_attn_probs=True)
    out = out.transpose(1,2)
    lse = lse.unsqueeze(-1)
    return out, lse
