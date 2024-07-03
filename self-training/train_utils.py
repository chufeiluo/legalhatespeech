from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM,
                          TrainingArguments, Trainer)
import torch
from pathlib import Path
from datasets import load_dataset
import numpy as np
import re
import pandas as pd
from tqdm.notebook import tqdm

import transformers
import datasets

from openprompt.data_utils import InputExample

from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback
#import pykeops
#from pykeops.torch import LazyTensor
from collections import Counter

from sklearn.metrics import f1_score, recall_score, precision_score, hamming_loss

from prompt_utils import preproc


class_mapping = {'Yes': 2, 'No': 0, 'Maybe': 1}

def customMetrics(ep):
    #custom softmax
    print(ep.predictions[0].shape)
    if len(ep.predictions) > 1:
        m = np.max(ep.predictions,axis=1,keepdims=True) #returns max of each row and keeps same dims
    else:
        m = np.max(ep.predictions[0])
    e_x = np.exp(ep.predictions - m) #subtracts each row with its max value
    s = np.sum(e_x,axis=1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / s 
    
    pred_round = np.round(f_x)
    print(pred_round, ep.label_ids)
    pred = pred_round == ep.label_ids[:len(pred_round)]
    
    negatives = np.logical_not(ep.label_ids[:len(pred)])
    guess = np.logical_and(pred, ep.label_ids[:len(pred)])
    not_cite = np.logical_and(pred, negatives)
    #print(np.sum(pred), np.sum(ep.label_ids[:len(pred)]), guess, not_cite)
    return {'tp': int(np.sum(guess.astype(float))),
           'fn': (int(np.sum(ep.label_ids[:len(pred)])) - int(np.sum(guess.astype(float)))),
            'tn': int(np.sum(not_cite.astype(float))),
           'fp': (int(np.sum(negatives)) - int(np.sum(not_cite.astype(float)))),
            'macro-f1': f1_score(ep.label_ids.astype(np.float16), pred_round, average='macro'),
            'macro-r': recall_score(ep.label_ids.astype(np.float16), pred_round, average='macro'),
            'macro-p': precision_score(ep.label_ids.astype(np.float16), pred_round, average='macro'),
            'micro-f1': f1_score(ep.label_ids.astype(np.float16), pred_round, average='micro'),
            'hamming': hamming_loss(ep.label_ids.astype(np.float16), pred_round)
           }


class TrainWrap:
    def __init__(self, model_ckpt, max_len=512, label_len=None, model=None, defs=[]):
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = model
        self.model_ckpt = model_ckpt
        
        self.max_len = max_len
        self.label_len = label_len
    
    def _process_labels(self, data, lab_name, labels=None, binary=False):

        return data
    
    
    def tokenize_and_encode_in(self, examples):
        return self.tokenizer(examples['post'], max_length=self.max_len, truncation=True, padding='max_length')
    
    def tokenize_and_encode_mlm(self, examples):
        return self.tokenizer(examples['masked'], max_length=self.max_len, truncation=True, padding='max_length', return_special_tokens_mask=True)
    
    def mlm_labels(self, examples):
        return {'labels': self.tokenizer(examples['post'], max_length=self.max_len, truncation=True, padding='max_length')['input_ids']}
    
    def convert_to_dataset(self, data, binary, prompt=False):
        data = Dataset.from_pandas(data, split="train")
        
        cols = data.column_names
        cols.remove('labels')
        if 'bin_lab' in cols:
            cols.remove('bin_lab')
        if 'label_mask' in cols:
            cols.remove('label_mask')

        print(f'tokenizing, prompt={prompt}, dataset: {data}')
        if prompt:
            _, data_enc = self.preprocess_mlm(data, 'post', labels=data['labels'])
        else:
            data_enc = data.map(self.tokenize_and_encode_in, batched=True, remove_columns=cols)
        #print(data_enc['labels'])
        
        print('converting labels')
        data_enc.set_format('torch')
        data_enc = (data_enc
              .map(lambda x : {"float_labels": x['labels'].to(torch.float)}, batched=True, remove_columns=['labels'])
              .map(lambda x : {"gate_labels": x['label_mask'].to(torch.float)}, batched=True, remove_columns=['label_mask'])
              .rename_column("float_labels", 'labels'))
        
        if binary:
            data_enc = (data_enc
              .map(lambda x : {"float_labels": x['bin_lab'].to(torch.float)}, batched=True, remove_columns=['bin_lab'])
              .rename_column("float_labels", 'bin_lab'))
        
        return [data, data_enc]
        
    def preprocess(self, data, lab_name, labels=None, prompt=False, inp_col='post', binary=False):
        #print(type(data[lab_name].iloc[0]))
        # creating labels
        if labels is not None: # pre-determined labels that need to be sorted
            if type(lab_name) is not list:
                lab_name = [lab_name]

            labs = []
            mask = []
            if type(data[lab_name[0]].iloc[0]) is np.float64: # labels are numerical
                for i in tqdm(range(len(data))):
                    tmp_lab = [0]*len(labels)
                    tmp_mask = [0]*len(labels)
                    for name in lab_name:
                        if np.isnan(data.iloc[i][name]):
                            tmp_lab[labels.index(name)] = 0
                        else:
                            tmp_lab[labels.index(name)] = round(data.iloc[i][name])
                            tmp_mask[labels.index(name)] = 1
                        #print(name)
                    labs.append(tmp_lab)
                    mask.append(tmp_mask)
            else: # labels are categorical, create based on absence/presence of label string
                possible_labels = set([x for y in data[lab_name[0]] for x in y])
                for x in tqdm(data[lab_name[0]]):
                    tmp_lab = [0]*len(labels)
                    tmp_mask = [0]*len(labels)
                    for i in range(len(labels)):
                        if labels[i] in x:
                            tmp_lab[i] = 1
                            tmp_mask[i] = 1
                        elif labels[i] in possible_labels:
                            tmp_mask[i] = 1

                    labs.append(tmp_lab)
                    mask.append(tmp_mask)
                #print(Counter(x for y in data[lab_name] for x in y), labels)
        else: # single target, categorical
            data[lab_name] = data[lab_name].astype('category')

            labels = data[lab_name].cat.categories

            data[lab_name] = data[[lab_name]].apply(lambda x: x.cat.codes)

            print(data[lab_name].value_counts(), labels)
            self.label_len = len(labels)
            labs = []
            for i in range(len(data[lab_name])):
                temp = [0]*len(data[lab_name].value_counts())
                temp[data[lab_name].iloc[i]] = 1

                labs.append(temp)
        
        if self.label_len is None or self.label_len != len(labels):
            self.label_len = len(labels)
        data['labels'] = labs
        data['label_mask'] = mask
        
        print('labels generated')
        
        if binary:
            data['bin_lab'] = [[np.max(x)] for x in data['labels']]
            print(data['bin_lab'])
        
        # cleaning input
        data['post'] = [re.sub(r'@.*?\s', 'USER ', x) for x in data[inp_col].to_list()]
        
        #print(labs)
        return self.convert_to_dataset(data, prompt=prompt, binary=binary)
    
    def preprocess_prompt(self, data, t, target, defs, mask_token='<mask>', include_def=True, text_name='post', id_col='id', open_prompt=True, use_labels=True):
        
        seqs, labels = preproc(data, t=t, target=target, defs=defs, include_def=include_def, text_name=text_name, open_prompt=open_prompt, labels=use_labels)
        
        
        if open_prompt:
            data_enc = [InputExample(guid=str(data.iloc[i%len(data)][id_col]), text_a=seqs[i][0], label=(class_mapping[labels[i]] if use_labels else -1), meta={'target': seqs[i][1], 'definition': seqs[i][2], 'target_groups': (seqs[i][3] if len(seqs[i]) == 4 else None)}) for i in range(len(labels))]
            
        else:
            masked = []
            for i in range(len(seqs)):
                masked.append(seqs[i] + mask_token)
                seqs[i] += labels[i]
                
            
            data = pd.DataFrame.from_dict({'post': seqs, 'masked': masked})
            
            data = Dataset.from_pandas(data, split="train")

            cols = data.column_names
            cols.remove('post')

            data_enc = data.map(self.tokenize_and_encode_mlm, batched=True, remove_columns=cols)
            data_enc = data_enc.map(self.mlm_labels, batched=True, remove_columns='post')
        return data, data_enc
    
    def preprocess_mlm(self, data, text_name, prompt_template=None, labels=None, mask_token='<mask>'):
        seqs = []
        masked = []
        for i in tqdm(range(len(data))):
            
            if prompt_template is not None:
                seqs.append(prompt_template.format(self.defs[labels[i]], data.iloc[i][text_name], labels[i]))
                masked.append(prompt_template.format(self.defs[labels[i]], data.iloc[i][text_name], mask_token))
            else:
                prompt = []
                #print(len(labels[i]), len(self.defs), data[text_name][i])
                for j in range(len(labels[i])):
                    if labels[i][j] == 1:
                        prompt.append(self.defs_template.format(self.defs[j][0], self.defs[j][1]))
                prompt = '\n'.join(prompt + [self.prompt])
                seqs.append(prompt.format(data[text_name][i]))
                masked.append(prompt.format(data[text_name][i]))
        
        data['post'] = seqs
        data['masked'] = masked
        
        data = Dataset.from_pandas(data, split="train")

        cols = data.column_names
        cols.remove('post')

        data_enc = data.map(self.tokenize_and_encode_mlm, batched=True, remove_columns=cols)
        data_enc = data_enc.map(self.mlm_labels, batched=True, remove_columns='post')
        
        return data, data_enc
            
    def train(self, run_name, train_enc, val_enc, test_enc=None,
              lr=2e-5, epochs=20, decay=0.01, batch_size=8, accum_steps=100, save_limit=5, device='cuda', best_model=True, metric_for_best_model='loss',
              callbacks=None, metrics=customMetrics, trainer_class=Trainer,
              ignore_data_skip=True,
              resume=False,
             dry_run=False,
             labels=['labels']):
        if self.model is None: # finetuning with huggingface
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_ckpt, num_labels=self.label_len).to(device)
            
        self.args = TrainingArguments(
                output_dir=run_name,
                evaluation_strategy = "steps",
                learning_rate=lr,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                weight_decay=decay,
                load_best_model_at_end=best_model,
                metric_for_best_model=metric_for_best_model,
                do_eval=True,
                logging_first_step=True,
                logging_steps=500,
                dataloader_num_workers=16,
                ignore_data_skip=ignore_data_skip,
                do_train=True,
                fp16=True,
                run_name = run_name,
                report_to = 'wandb',
                label_names=labels,
                eval_accumulation_steps=accum_steps,
                save_total_limit = save_limit, # Only last 5 models are saved. Older ones are deleted.
            )
        
        trainer = trainer_class(model=self.model, 
                  args=self.args, 
                  train_dataset=train_enc, 
                  eval_dataset=val_enc, 
                  tokenizer=self.tokenizer, 
                  compute_metrics=metrics,
                  callbacks=callbacks
                 )
        self.trainer = trainer
        if dry_run:
            print(self.trainer.evaluate())
        trainer.train(resume_from_checkpoint=resume)
        if test_enc is not None:
            print(trainer.predict(test_enc))
        return trainer.model
    
    def fewshot(self, run_name, train_enc, val_enc, model, test_enc=None,
              lr=2e-5, steps=32, decay=0.01, batch_size=8, accum_steps=100, save_limit=5, device='cuda', best_model=False, metric_for_best_model='loss',
              callbacks=None, metrics=customMetrics, trainer_class=Trainer,
              resume=False):
        
        
        self.args = TrainingArguments(
                output_dir=run_name,
                evaluation_strategy = "steps",
                learning_rate=lr,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                max_steps=steps,
                weight_decay=decay,
                load_best_model_at_end=best_model,
                metric_for_best_model=metric_for_best_model,
                do_eval=True,
                logging_first_step=True,
                logging_steps=steps,
                dataloader_num_workers=16,
                do_train=True,
                fp16=True,
                run_name = run_name,
                report_to = 'wandb',
                eval_accumulation_steps=accum_steps,
                save_total_limit = save_limit, # Only last 5 models are saved. Older ones are deleted.
            )
        
        trainer = trainer_class(model=model, 
                  args=self.args, 
                  train_dataset=train_enc, 
                  eval_dataset=val_enc, 
                  tokenizer=self.tokenizer, 
                  compute_metrics=metrics,
                  callbacks=callbacks
                 )
        self.trainer = trainer
        print(self.trainer.evaluate())
        trainer.train(resume_from_checkpoint=resume)
        if test_enc is not None:
            print(trainer.predict(test_enc))
        return trainer.model
    
    def evaluate(self, test_enc):
        if model is None: # finetuning with huggingface
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_ckpt).to(device)
        predictions = []
        
        test_dl = DataLoader(test_enc)
        
        with torch.no_grad():
            for batch in test_dl:
                
                output = model(**batch)
                
     
        
        

        