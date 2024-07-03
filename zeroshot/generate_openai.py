import os
from tqdm.notebook import tqdm
import openai

import time, json
import tiktoken
from datasets import load_dataset
import re, html

from argparse import ArgumentParser


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


def generate(instruction, inp, model_name, max_len=4096, reasoning=False):
    outp = None
    message = [{'role': 'system', 'content': instruction}]
    enc = tiktoken.encoding_for_model(model_name)

    if len(enc.encode(instruction +  inp)) > max_len:
        prompt_len = len(enc.encode(instruction))
        inp = enc.decode(enc.encode(inp)[:max_len-prompt_len])
    if reasoning:
        message.append({'role': 'user', 'content': inp.split('. A:')[0] + 'If Yes, explain why.'})
    else:
        message.append({'role': 'user', 'content': inp})
    
    while outp is None:
        try:
            result = openai.ChatCompletion.create(
                model=model_name,
                messages=message,
                max_tokens=100,
                top_p = 0.1
            )

            outp = result['choices'][0]['message']['content']

            usage = result['usage']
        except Exception as e:
            print(e)
            outp = None
            time.sleep(5)
    
    return outp, usage

    
    

# To get the tokeniser corresponding to a specific model in the OpenAI API:
def main(data, args):
    results = {}
    total_cost = 0

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
        
        while i < len(v['train']):
            print(k, total_cost)

            res, usage = generate(v['train'][i]['instruction'], v['train'][i]['input'], args.model_name, reasoning=args.reasoning)
            sequences.append(res)

            total_cost += ((usage['prompt_tokens']/1000)*0.0015)+((usage['completion_tokens']/1000)*0.002)
            with open(os.path.join(cache, '{0}.jsonl'.format(k)), 'a') as f:
                f.write(json.dumps({'generation': res, 'id': i}))
                f.write('\n')

        results[k] = sequences

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
    parser.add_argument("--dataset_dir", type=str, default='data')
    args = parser.parse_args()

    data_files = os.listdir(args.dataset_dir)

    data = {}

    for file in data_files:
        if file.endswith('json') and 'nonotes' in file:
            data[file] = load_dataset('json', data_files=os.path.join(args.dataset_dir, file))
            
    main(data, args)