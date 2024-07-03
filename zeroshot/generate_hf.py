import os
from tqdm.notebook import tqdm
import openai

import time, json
import tiktoken
from datasets import load_dataset
import re, html

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from argparse import ArgumentParser
# >>> dataset = load_dataset('json', data_files='my_file.json')





enc = tiktoken.encoding_for_model("gpt-4-0613")
results = {}
def clean_text(text):
    inp = re.sub(r'(u/.*? |@.*?( |$))', "<user> ", text)
    inp = re.sub(r'\u2019', "'", inp)
    inp = re.sub(r'[“”]', '"', inp)
    inp = re.sub(r'\\u[a-f0-9]{4}', "'", inp)
    inp = re.sub(r'(https://[a-zA-Z.-_?]*)', "(URL)", inp)

    inp = re.sub(r'[\s]{2,}', " ", inp)
    inp = html.unescape(inp).encode('ascii', errors='ignore').decode()
    return inp

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]   

# To get the tokeniser corresponding to a specific model in the OpenAI API:
def main(data, args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    results = {}

    cache = os.path.join(args.cache_dir, re.sub(r'[^a-zA-Z0-9]', '_', args.model_name))
    os.makedirs(cache, exist_ok=True)

    for k, v in data.items():
        i = 0

        sequences = []
        
        if os.path.exists(os.path.join(cache, '{0}.jsonl'.format(k))):
            with open(os.path.join(cache, '{0}.jsonl'.format(k)).format(k), 'r') as f:
                for line in f.readlines():
                    try:
                        temp = json.loads(line)
                        if temp['id'] == i+1:
                            sequences.append(temp['generation'])
                            i += 1
                    except Exception as e:
                        pass
                print(i)
                assert i == len(sequences)-1
                i += 1
        
        test_prompt = [clean_text('<s>[INST]' + x['instruction']  + enc.decode(enc.encode(x['input'])[-400:])) + 'If Yes, explain why.' + '[\INST]\n' for x in v['train']]
        
        for b in batch(test_prompt):
            seq = pipeline(
                b,
                max_new_tokens=128,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
            answers = [seq[i][0]['generated_text'] for i in range(len(seq))]

            results[k].extend(answers)

            with open(os.path.join(cache, '{0}.jsonl'.format(k)), 'a') as f:
                f.write('\n'.join([json.dumps({'generation': answers[i], 'id': len(results[k]) + i}) for i in range(len(answers))]))
                f.write('\n')

    with open(os.path.join(cache, 'final.json'), 'w') as f:
        f.write(json.dumps(results))
    
    
if __name__ == '__main__': # if run directly, test the entire dataset
    parser = ArgumentParser()

    parser.add_argument("--output_dir", type=str)

    # Model params
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument('--reasoning', action='store_true')

    # Dataset params
    parser.add_argument("--dataset_dir", type=str, required=True)
    args = parser.parse_args()

    data_files = os.listdir(args.dataset_dir)

    data = {}

    for file in data_files:
        if file.endswith('json') and 'nonotes' in file:
            data[file] = load_dataset('json', data_files=os.path.join(args.dataset_dir, file))
            
    main(data, args)