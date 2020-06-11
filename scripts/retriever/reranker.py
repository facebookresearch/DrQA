import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from tqdm import tqdm, trange

from question_classifier.input_example import InputExample
from transformers import (
        WEIGHTS_NAME,
        AdamW,
        AlbertConfig,
        AlbertForSequenceClassification,
        AlbertTokenizer,
        BertConfig,
        BertForSequenceClassification,
        BertTokenizer,
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
        RobertaConfig,
        RobertaForSequenceClassification,
        RobertaTokenizer,
        XLMConfig,
        XLMForSequenceClassification,
        XLMRobertaConfig,
        XLMRobertaForSequenceClassification,
        XLMRobertaTokenizer,
        XLMTokenizer,
        XLNetConfig,
        XLNetForSequenceClassification,
        XLNetTokenizer,
        get_linear_schedule_with_warmup,
        )

from transformers import glue_convert_examples_to_features as convert_examples_to_features

class Reranker:

    def __init__(self, model_type, model_path, max_seq, batch_size=256):
        self.model_type = model_type
        self.model_path = model_path
        self.max_seq = max_seq
        self.batch_size = batch_size

    def load_model(self):
        # load fined tuned model 

        MODEL_CLASSES = {
                'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
                'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
                'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
                "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
                }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.model_type] 

        self.model = model_class.from_pretrained(self.model_path)
        self.tokenizer = tokenizer_class.from_pretrained(self.model_path)

        self.model.to(device)


    def __getPredictions(self, dataloader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        preds = []
        Softmax = torch.nn.Softmax(dim=1)

        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], 'labels':None}
                if self.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                            batch[2] if self.model_type in ["bert", "xlnet", "albert"] else None
                            )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            
                outputs = self.model(**inputs)
            
            logits = outputs[0]
            logits = Softmax(logits) 
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)


        preds = preds[0]
        preds = [pred for (pred,_) in preds]
        return preds

    def __transform_to_features(self, samples):

        features = convert_examples_to_features(samples, 
                self.tokenizer, 
                self.max_seq,
                label_list=['answerable','not_answerable'], 
                output_mode='classification', 
                pad_on_left=bool(self.model_type in ["xlnet"]),
                pad_token=self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0],
                pad_token_segment_id=4 if self.model_type in ["xlnet"] else 0,
                )
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_types_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        data = TensorDataset(all_input_ids, all_attention_mask, all_token_types_ids, all_labels)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader

    def evaluate(self, samples):
        dl = self.__transform_to_features(samples)
        return self.__getPredictions(dl)
