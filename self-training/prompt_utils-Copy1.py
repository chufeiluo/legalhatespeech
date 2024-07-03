
import numpy as np
import re, random
import pandas as pd
from tqdm import *

from collections import Counter

from sklearn.metrics import f1_score, recall_score, precision_score, hamming_loss

from ast import literal_eval


legal_defs = pd.read_csv('legal definitions.csv')

def_template = '''{0} is defined as {1}.'''
protected_groups = '''Protected/identifiable groups include {0}'''
example_template = '''Not a violation: {0}\nViolation: {1}'''
prompt_template = '''Post: {0}\nQ: Is this post a violation of {1}? Please answer yes or no. A: {2}'''

def sbic_labels(data):
    labs = []
    
    for i in tqdm(range(len(data))):
        if (data.iloc[i]['intentYN'] != 0) & (data.iloc[i]['offensiveYN'] != 0) & (data.iloc[i]['hasBiasedImplication'] == 0):
            labs.append('Yes')
        else:
            labs.append('No')
    return labs


def incitement_labels(data, target):
    labs = []
    label = 'implicit_class'
    
    for i in tqdm(range(len(data))):
        if data.iloc[i][label] in target:
            labs.append('Yes')
        else:
            labs.append('No')
            
    return labs

def preproc(data, prompt, label_gen, tw):

    labs = label_gen(data)
    return tw.preprocess_mlm(data, 'post', prompt, labs)

def continuous_labels(data, label):
    labs = []
    
    #label = 'labels'
    
    for i in tqdm(range(len(data))):
        if data.iloc[i][label] <= 0.2:
            labs.append('No')
        elif data.iloc[i][label] >= 0.8:
            labs.append('Yes')
        else:
            labs.append('Maybe')
    return labs

def hatespeech_labels(data, label):
    labs = []
    
    #label = 'labels'
    
    for i in tqdm(range(len(data))):
        if data.iloc[i][label] == 0:
            labs.append('No')
        elif data.iloc[i][label] == 1:
            labs.append('Yes')
        else:
            labs.append('Maybe')
    return labs


def finegrained(data, target):
    labs = []
    label = 'finegrained'
    
    for i in tqdm(range(len(data))):
        if target.lower() in data.iloc[i][label].lower():
            labs.append('Yes')
        else:
            labs.append('No')
            
    return labs

def prompt_formatting(data, text_name, labs, fewshot, target, defs, examples=None, include_def=True):
    inp = []
    prompt_name = 'hate speech'
    for i in tqdm(range(len(data))):
        temp = []
        if include_def:
            temp.append(def_template.format(defs[defs['name'] == target].iloc[0]['promptName'], defs[defs['name'] == target].iloc[0]['definition']))
            prompt_name = defs[defs['name'] == target].iloc[0]['promptName']
        
        if 'protected_groups' in defs.columns:
            temp.append(protected_groups.format(defs[defs['name'] == target].iloc[0]['protected_groups']))
        
        if fewshot:
            for e in examples:
                temp.append(prompt_template.format(e[0], target, e[1]))
        
        temp.append(prompt_template.format(data.iloc[i][text_name], prompt_name, ''))
        
        inp.append('\n'.join(temp))
        
    return inp
        
def preproc(data, t='finegrained', target='CC_318', fewshot=False, indices=None, defs=legal_defs, include_def=True, text_name='post', open_prompt=True, labels=True):
    
    examples=None
    
    seqs = None
    
    if t=='finegrained':
        if type(target) is list:
            labs = []
            seqs = []
            for tar in target:
                name = defs[defs['name'] == tar]['promptName'].iloc[0]
                definition = defs[defs['name'] == tar]['definition'].iloc[0]
                groups = defs[defs['name'] == tar]['protected_groups'].iloc[0]
                lab = (finegrained(data, tar) if labels else ['']*len(data))
                labs.extend(lab)
                if not open_prompt:
                    seqs.extend(prompt_formatting(data, text_name, lab, fewshot, tar, defs, examples, include_def))
                else:
                    seqs.extend([(x, name, definition, groups) for x in data[text_name].to_list()])
        else:
            labs = (finegrained(data, target) if labels else ['']*len(data))
    elif t == 'hatespeech':
        if type(target) is list:
            labs = []
            seqs = []
            for tar in target:
                name = defs[defs['name'] == tar]['promptName'].iloc[0]
                definition = defs[defs['name'] == tar]['definition'].iloc[0]
                groups = defs[defs['name'] == tar]['protected_groups'].iloc[0]
                lab = (hatespeech_labels(data, 'labels') if labels else ['']*len(data))
                labs.extend(lab)
                if not open_prompt:
                    seqs.extend(prompt_formatting(data, text_name, lab, fewshot, tar, defs, examples, include_def))
                else:
                    seqs.extend([(x, name, definition, groups) for x in data[text_name].to_list()])
        else:
            labs = (finegrained(data, target) if labels else ['']*len(data))
    elif t == 'continuous':
        if type(target) is list:
            labs = []
            seqs = []
            for tar in target:
                
                name = defs[defs['name'] == tar]['promptName'].iloc[0]
                definition = defs[defs['name'] == tar]['definition'].iloc[0]
                
                lab = (continuous_labels(data, tar)  if labels else ['']*len(data))
                labs.extend(lab)
                if not open_prompt:
                    seqs.extend(prompt_formatting(data, text_name, lab, fewshot, tar, defs, examples, include_def))
                else:
                    seqs.extend([(x, name, definition) for x in data[text_name].to_list()])
        else:
            labs = (continuous_labels(data, target) if labels else ['']*len(data))
            
    elif t == 'sbic':
        labs = sbic_labels(data)
    elif t == 'implicit':
        labs = []
        seqs = []
        for tar in target:
            name = defs[defs['name'] == tar]['promptName'].iloc[0]
            definition = defs[defs['name'] == tar]['definition'].iloc[0]
            lab = (incitement_labels(data, tar)  if labels else ['']*len(data))
            labs.extend(lab)
            if not open_prompt:
                seqs.extend(prompt_formatting(data, text_name, lab, fewshot, tar, defs, examples, include_def))
            else:
                seqs.extend([(x, name, definition) for x in data[text_name].to_list()])
    
    if fewshot:
        if indices==None:
            indices = random.sample(range(len(data)), 3)
        
        examples = [(data[i], labs[i]) for i in indices]
        data = data[~data.index.isin(indices)]
        data.reset_index(inplace=True)
        
    
    if seqs is None:
        if not open_prompt:
            seqs = prompt_formatting(data, text_name, labs, fewshot, target, defs, examples, include_def)
        else:
            name = defs[defs['name'] == target]['promptName'].iloc[0]
            definition = defs[defs['name'] == target]['definition'].iloc[0]
            seqs = [(x, name, definition) for x in data[text_name].to_list()]        
    return [seqs, labs]


def evaluate(outputs, labs, target):
    print(target)
    top_ans = []
    for o in outputs:
        try:
            if 'yes' in o.lower() or 'violates' in o.lower():
                top_ans.append(1)
            elif 'no' in o.lower() or 'does not violate' in o.lower():
                top_ans.append(0)
            else:
                print(o)
                #print(o[0]['generated_text'].lower().split('a:')[-1])
                top_ans.append(-1)
        except Exception as e:
            print(o)
            top_ans.append(-1)
    print(Counter(top_ans))
#     labs = []
#     for x in test['finegrained'].to_list():
#         #print(literal_eval(x))
#         labs.append(target in literal_eval(x))

    if type(labs[0]) is str:
        labs = [(1 if x.lower() == 'yes' else 0) for x in labs]
    
    bad_results = []
    print([(top_ans[i], labs[i]) for i in range(len(top_ans))])
    for i in range(len(top_ans)):
        if top_ans[i] != labs[i]:
            bad_results.append(outputs[i])
    
    print(Counter(labs))

    return [(f1_score(labs, [(x if x != -1 else 0) for x in top_ans]),
         precision_score(labs, [(x if x != -1 else 0) for x in top_ans]),
         recall_score(labs, [(x if x != -1 else 0) for x in top_ans])), bad_results]