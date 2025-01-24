
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig, GPT2Tokenizer
from transformers import GPTNeoXTokenizerFast
from open_lm.model import create_model, create_params

from typing import Callable
from open_lm.attention import get_attn_func, xformers_attn, torch_attn

import yaml
from yaml import Loader

from open_lm.params import add_model_args
import wandb

from quest.replace_llm_attention import patch_attention_layers

class ModelArgs:
    def __init__(self, path: str):
        with open(path, "r") as f:
            params = yaml.load(f, Loader=Loader)
        for k, v in params.items():
            setattr(self, k, v)

def load_encoded_texts(tokenizer, args):
    # Load LongBench datasets
    dataset = 'longbench'
    dataset_names = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    data_subset_all = []
    for dataset in dataset_names:
        data_ = load_dataset('THUDM/LongBench', f"{dataset}", split='test')
        data_subset = data_.filter(lambda x: len(tokenizer.encode(x['context'])) >= args.seq_len)
        if len(data_subset) > 0:
            data_subset_all.append(data_subset)
    data = concatenate_datasets(data_subset_all)

    encoded_texts = []
    pbar = tqdm(data)
    for i, data_i in enumerate(pbar):
        encoded_text = tokenizer.encode(data_i['context'], return_tensors='pt', truncation=True)
        pbar.set_description(f"seq_len: {len(encoded_text[0])}, n_data: {len(encoded_texts)}")
        if len(encoded_text[0]) < args.seq_len:
            continue
        encoded_texts.append(encoded_text)
    print(f"# of data longer than {args.seq_len}: {len(encoded_texts)}")
    return encoded_texts

@torch.no_grad()
def perplexity_eval(model, encoded_texts, args, device):
    loss_fct = CrossEntropyLoss(reduction="none")
    pbar = tqdm(range(len(encoded_texts)))
    ppls = []
    dtype = torch.bfloat16
    for bid in pbar:
        encoded_batch = encoded_texts[bid:bid+1]
        if type(encoded_batch) == dict:
            attn_mask = encoded_batch['attention_mask'] if 'attention_mask' in encoded_batch.keys() else None
            encoded_batch = encoded_batch['input_ids']
        elif type(encoded_batch) == list:
            encoded_batch = encoded_batch[0]
        
        encoded_batch = encoded_batch.to(device)
        attn_mask = torch.ones_like(encoded_batch)

        print(f"encoded_batch: {encoded_batch.shape}, attn_mask: {attn_mask.shape}")
        # Logits is the first output with length of vocab size
        out_logits = model(encoded_batch)[0]
        # out_logits = model(encoded_batch).logits

        labels = encoded_batch

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        loss_ = loss_fct(shift_logits.transpose(1, 2), shift_labels).float()
        perplexity_batch = torch.exp2(
            (loss_ * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )
        ppls += perplexity_batch.tolist()

        pbar.set_description(f"[{bid:<4}/{len(encoded_texts)}] avg_ppls: {np.mean(np.array(ppls)[~np.isnan(np.array(ppls))]):.4f}")
        
        del out_logits, encoded_batch, attn_mask, shift_logits, shift_labels, shift_attention_mask_batch, perplexity_batch
        

    nan_cnt = sum(np.isnan(np.array(ppls)))
    ppl_mean = np.mean(np.array(ppls)[~np.isnan(np.array(ppls))])

    print(f"ppl: {ppl_mean}, nan_cnt: {nan_cnt}")
    res_str = f"model: {args.model_name}, dtype: {dtype}, seq_len: {args.seq_len}, num_patch_layers: {args.num_patch_layers}, n_data: {len(encoded_texts)}, ppl: {ppl_mean}, nan_cnt: {nan_cnt}\n"
    print(res_str)


def get_model_and_tokenizer(model_name, args):

    if model_name == "chatglm2-6b-32k":
        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("THUDM/chatglm2-6b-32k", trust_remote_code=True)

    elif model_name == "glm-edge-1.5b-chat":
        tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-edge-1.5b-chat", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("THUDM/glm-edge-1.5b-chat", trust_remote_code=True)

    elif model_name == "opt-350m-50k":
        tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
        config = AutoConfig.from_pretrained("facebook/opt-350m")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", config=config, torch_dtype=torch.float16, device_map="auto")

    elif model_name == 'open_lm_1b':
        checkpoint = torch.load('/home/lj9979/QUEST/models/open_lm_1b.pt')
        model = create_model(args).half()
        state_dict = checkpoint["state_dict"]
        state_dict = {x.replace("module.", ""): y for x, y in state_dict.items()}
        model.load_state_dict(state_dict)
        tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
        model.to(device="cuda", dtype=torch.bfloat16)
        model.eval()
        print("performing perplexity evaluation on open_lm_1b")
        encoded_texts = load_encoded_texts(tokenizer, args)
        perplexity_eval(model.to("cuda"), encoded_texts, args, device="cuda")
    else:
        raise NotImplementedError("Currently we only support chatglm2")
            
    return model, tokenizer

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=32768)
    # patch config
    parser.add_argument("--patch_config", type=str, default="odd", choices=['last', 'first', 'even', 'odd'])
    parser.add_argument("--attn_method", type=str, default="prefill", choices=['flash', 'hyper', 'hyper-cuda', 'prefill'])
    parser.add_argument("--num_patch_layers", type=int, default=-1)
    # params of HyperAttention
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--sample_size", type=int, default=256)
    parser.add_argument("--lsh_num_projs", type=int, default=7)
    parser.add_argument("--min_seq_len", type=int, default=4096)
    # currently only supports **chatglm2-6b-32k**
    parser.add_argument("--model_name", type=str, default="open_lm_1b", choices=['open_lm_1b', 'glm-edge-1.5b-chat', 'chatglm2-6b-32k', 'opt-350m-32k'])
    # args for the open_lm model
    parser.add_argument("--checkpoint", default="/home/lj9979/QUEST/models/open_lm_1b.pt")
    # TODO: Make this take as input --model-config, similar to generate.py
    parser.add_argument("--params", default="llm")
    parser.add_argument("--wandb-dir", default="")
    # parser.add_argument("--input-text", required=True)
    parser.add_argument("--max-gen-len", default=200, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--model", default="/home/lj9979/QUEST/models/open_lm_1b.json")
    if parser.parse_args().model_name == 'open_lm_1b':
        add_model_args(parser)
        param_config = create_params(parser.parse_args())
        parser.add_argument("--param_config", default= param_config)

    return parser.parse_args()

@torch.no_grad()
def main():
    args = get_arguments()
    
    for arg_name, arg_var in args.__dict__.items():
        print(f"{arg_name:<16} : {arg_var}")

    model, tokenizer = get_model_and_tokenizer(args.model_name, args)
    tokenizer.model_max_length = args.seq_len
    device = "cuda"
    dtype = torch.bfloat16

    # Load LongBench datasets
    dataset = 'longbench'
    dataset_names = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
        "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
        "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    
    data_subset_all = []
    for dataset in dataset_names:
        data_ = load_dataset('THUDM/LongBench', f"{dataset}", split='test')
        data_subset = data_.filter(lambda x: len(tokenizer.encode(x['context'])) >= args.seq_len)
        if len(data_subset) > 0:
            data_subset_all.append(data_subset)
    data = concatenate_datasets(data_subset_all)

    encoded_texts = []
    pbar = tqdm(data)
    for i, data_i in enumerate(pbar):
        encoded_text = tokenizer.encode(data_i['context'], return_tensors='pt', truncation=True)
        pbar.set_description(f"seq_len: {len(encoded_text[0])}, n_data: {len(encoded_texts)}")
        if len(encoded_text[0]) < args.seq_len:
            continue
        encoded_texts.append(encoded_text)
    print(f"# of data longer than {args.seq_len}: {len(encoded_texts)}")
    
    if args.attn_method != 'flash':
        patch_attention_layers(model_config=model, **args.__dict__)

    model.to(device=device, dtype=dtype)
    model.eval()
    loss_fct = CrossEntropyLoss(reduction="none")

    ppls = []

    pbar = tqdm(range(len(encoded_texts)))
    for bid in pbar:
        encoded_batch = encoded_texts[bid:bid+1]
        if type(encoded_batch) == dict:
            attn_mask = encoded_batch['attention_mask'] if 'attention_mask' in encoded_batch.keys() else None
            encoded_batch = encoded_batch['input_ids']
        elif type(encoded_batch) == list:
            encoded_batch = encoded_batch[0]
        
        encoded_batch = encoded_batch.to(device)
        attn_mask = torch.ones_like(encoded_batch)

        print(f"encoded_batch: {encoded_batch.shape}, attn_mask: {attn_mask.shape}")
        # Logits is the first output with length of vocab size
        out_logits = model(encoded_batch)[0]
        # out_logits = model(encoded_batch).logits

        labels = encoded_batch

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        loss_ = loss_fct(shift_logits.transpose(1, 2), shift_labels).float()
        perplexity_batch = torch.exp2(
            (loss_ * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )
        ppls += perplexity_batch.tolist()

        pbar.set_description(f"[{bid:<4}/{len(encoded_texts)}] avg_ppls: {np.mean(np.array(ppls)[~np.isnan(np.array(ppls))]):.4f}")
        
        del out_logits, encoded_batch, attn_mask, shift_logits, shift_labels, shift_attention_mask_batch, perplexity_batch
        

    nan_cnt = sum(np.isnan(np.array(ppls)))
    ppl_mean = np.mean(np.array(ppls)[~np.isnan(np.array(ppls))])

    print(f"ppl: {ppl_mean}, nan_cnt: {nan_cnt}")
    res_str = f"model: {args.model_name}, dtype: {dtype}, seq_len: {args.seq_len}, num_patch_layers: {args.num_patch_layers}, n_data: {len(encoded_texts)}, ppl: {ppl_mean}, nan_cnt: {nan_cnt}\n"
    print(res_str)

if __name__ == "__main__":
    main()
