import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm, trange

from question_classifier.input_example import InputExample
from question_classifier import utils as bert_utils
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

class Reranker:

    def __init__(self, model_type, model_path, max_seq):
        self.model_type = model_type
        self.model_path = model_path
        self.max_seq = max_seq


    def load_model(self):
        # load fined tuned model 

        MODEL_CLASSES = {
                'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
                'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
                'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
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
        Softmax = torch.nn.Softmax(1)

        for input_ids, input_mask, segment_ids, label_ids in dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask, labels=None)[0]
                logits = Softmax(logits) 
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)


        preds = preds[0]
        preds = [preds[i,1] for i in range(len(preds))]
        return preds

    def __transform_to_features(self, samples):

        features = bert_utils.convert_examples_to_features(samples,['not_answerable', 'answerable'], self.max_seq, self.tokenizer,'classification', 
                cls_token_at_end=bool(self.model_type in ['xlnet']),
                cls_token=self.tokenizer.cls_token,
                sep_token=self.tokenizer.sep_token,
                cls_token_segment_id=2 if self.model_type in ['xlnet'] else 0,
                pad_on_left=bool(self.model_type in ['xlnet']),
                pad_token_segment_id=4 if self.model_type in ['xlnet'] else 0)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader

    def evaluate(self, samples):
        dl = self.__transform_to_features(samples)
        return self.__getPredictions(dl)
