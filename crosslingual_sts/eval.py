import re
import sys
import io, os
import torch
import numpy as np
import logging
import tqdm
import fcntl
import time
import argparse
import transformers
from transformers import LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_embedding_sentence', action='store_true')
    parser.add_argument('--mask_embedding_sentence_template', type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str,
                        help="Transformers' model name or path")
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test',
                        help="What evaluation mode to use (dev: fast mode, dev results; test: full mode, test results); fasttest: fast mode, test results")
    parser.add_argument('--load_kbit', type=int,
                        choices=[32],
                        default=32,
                        help="Load model in kbit")

    parser.add_argument('--avg', action='store_true', help="Average pooling")
    parser.add_argument('--lora_weight', type=str, default=None)
    parser.add_argument('--tensor_parallel', action='store_true')


    args = parser.parse_args()

    # Load the model
    if args.tensor_parallel:
        # Use 16bit if employing tensor parrallel
        import tensor_parallel as tp
        n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                        low_cpu_mem_usage = True, torch_dtype=torch.float16)
        model = tp.tensor_parallel(model, [i for i in range(n_gpus)])
    elif args.load_kbit == 4:
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            load_in_4bit=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            torch_dtype=torch.float16,
            device_map='auto',
        )
    elif args.load_kbit == 8:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     device_map='auto',
                                                     output_hidden_states=True,
                                                     trust_remote_code=True,
                                                     load_in_8bit=args.load_kbit == 8,)
    elif args.load_kbit == 32:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     device_map='auto',
                                                     output_hidden_states=True,
                                                     trust_remote_code=True,
                                                     torch_dtype=torch.float32)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                     device_map='auto',
                                                     output_hidden_states=True,
                                                     trust_remote_code=True,
                                                     torch_dtype=torch.float16)

    if args.lora_weight is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model,
            args.lora_weight,
            torch_dtype=torch.float16,
            device_map={'': 0},
        )
        if args.load_kbit == 4:
            from peft.tuners.lora import LoraLayer
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    #module = module.to(torch.bfloat16)
                    module = module.to(torch.float16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        #module = module.to(torch.bfloat16)
                        if 'opt' in args.model_name_or_path:
                            module = module.to(torch.float32)
                        else:
                            module = module.to(torch.float16)

    # Load the tokenizer
    if 'llama' in args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        tokenizer.bos_token_id = 1
        tokenizer.eos_token = '</s>'
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set up the tasks
    args.tasks = ['STS17']

    # Set params for SentEval
    if args.mode == 'dev' or args.mode == 'fasttest':
        # Fast mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5, 'batch_size': 32}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 32, 'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        # Full mode
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10, 'batch_size':16}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        input_sentences = [' '.join(s) for s in batch]
        if max_length == 500:
            sentences = [tokenizer.decode(tokenizer.encode(s, add_special_tokens=False)[:max_length]) for s in sentences]
            max_length = 512

        if args.mask_embedding_sentence and args.mask_embedding_sentence_template is not None:
            # *cls*_This_sentence_of_"*sent_0*"_means*mask*.*sep+*
            template = args.mask_embedding_sentence_template
            template = template.replace('_', ' ').replace('*sep+*', '')\
                                                    .replace('*cls*', '')

            for i, s in enumerate(sentences):
                if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
                s = s.replace('"', '\'')
                if len(s) > 0 and '?' == s[-1]: s = s[:-1] + '.'
                sentences[i] = template.replace('*sent 0*', s).strip()

        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
            max_length=max_length,
            truncation=max_length is not None
        )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device) if batch[k] is not None else None

        # Get raw embeddings
        with torch.no_grad():
            hidden_states = model(output_hidden_states=True, return_dict=True, **batch).hidden_states
            if args.avg:
                last_layer = hidden_states[-1]
                attention_mask = batch['attention_mask'].unsqueeze(-1).expand(last_layer.shape)
                outputs = (last_layer * attention_mask).mean(1)
            else:
                outputs = hidden_states[-1][:, -1, :]

            if outputs.dtype == torch.bfloat16:
                # bfloat16 not support for .numpy()
                outputs = outputs.float()

            return outputs.cpu()

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

if __name__ == "__main__":
    main()
