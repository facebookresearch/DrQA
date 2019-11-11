#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate the accuracy of the DrQA retriever module."""

import regex as re
import logging
import argparse
import json
import time
import os
import numpy as np

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from drqa import retriever, tokenizers
from drqa.retriever import utils
from question_classifier.input_example import InputExample
from question_classifier import utils as bert_utils

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                                      TensorDataset)
from tqdm import tqdm, trange
import logging
from torch.nn import CrossEntropyLoss, MSELoss
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
        BertForSequenceClassification, BertTokenizer,
        XLMConfig, XLMForSequenceClassification,
        XLMTokenizer, XLNetConfig,
        XLNetForSequenceClassification, XLNetTokenizer)

logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------
# Multiprocessing target functions.
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None


def init(tokenizer_class, tokenizer_opts, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


def has_answer(answer, doc_id, match):
    """Check if a document contains an answer string.

    If `match` is string, token matching is done between the text and answer.
    If `match` is regex, we search the whole text with the regex.
    """
    global PROCESS_DB, PROCESS_TOK
    text = PROCESS_DB.get_doc_text(doc_id)
    text = utils.normalize(text)
    if match == 'string':
        # Answer is a list of possible strings
        text = PROCESS_TOK.tokenize(text).words(uncased=True)
        for single_answer in answer:
            single_answer = utils.normalize(single_answer)
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i + len(single_answer)]:
                    return True
    elif match == 'regex':
        # Answer is a regex
        single_answer = utils.normalize(answer[0])
        if regex_match(text, single_answer):
            return True
    return False

def getPredictions(samples, model, tokenizer, max_seq, doc_lengths, args):
    features = bert_utils.convert_examples_to_features(samples,['not_answerable', 'answerable'], max_seq, tokenizer,'classification', 
            cls_token_at_end=bool(args.model_type in ['xlnet']),
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            pad_on_left=bool(args.model_type in ['xlnet']),
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    preds = []
    Softmax = torch.nn.Softmax(1)
    for input_ids, input_mask, segment_ids, label_ids in dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)[0]
            logits = Softmax(logits) 
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
    
    preds = preds[0]
    preds = [preds[i,1] for i in range(len(preds))]
    ret=[]
    begin = 0
    end = 0

    print(doc_lengths)
    for i in doc_lengths:
        end+=i
        ret.append(np.amax(preds[begin:end]))
        begin+=i
   
    return ret

def init_classifier(path, args):
    # load fined tuned model 

    MODEL_CLASSES = {
                'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
                'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
                'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
                }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type] 
    model = model_class.from_pretrained(path)
    tokenizer = tokenizer_class.from_pretrained(path)
    
    model.to(device)
    
    return model, tokenizer

def rerankDocs(questions, answers, closest_docs, db, classifier, max_seq, args):

    uuid=0
    model, tokenizer = init_classifier(classifier, args)
    documents = []
    for doc_ids, _ in closest_docs:
        batch = []
        for doc_id in doc_ids:
            text = db.get_doc_text(doc_id)
            batch.append((utils.normalize(text), doc_id))
        documents.append(batch)
    
    return_documents = []

    for question, docs in tqdm(zip(questions, documents)):
        len_question = len(tokenizer.tokenize(question))
        samples = []
        doc_lengths = []
        for doc, doc_id in docs:
            paragraphs = doc.strip("\n\n\n")
            paragraphs = paragraphs.split("\n\n")

            contexts = []
            to_add = ""
            for temp in paragraphs:
                len_temp = len(tokenizer.tokenize(temp))
                if len_temp > (max_seq - len_question):
                    if len(to_add) > 1:
                        contexts.append(InputExample(guid=uuid, text_a=to_add, text_b=question, label="not_answerable"))
                        uuid+=1
                        to_add = ""
                    contexts.append(InputExample(guid=uuid, text_a=temp, text_b=question, label="not_answerable"))
                    uuid+=1
                elif len(tokenizer.tokenize(to_add)) + len_temp <= (max_seq - len_question):
                    to_add = to_add + temp
                else:
                    contexts.append(InputExample(guid=uuid, text_a=to_add, text_b=question, label="not_answerable"))
                    uuid+=1
                    to_add = temp
            if len(to_add) > 1:
                contexts.append(InputExample(guid=uuid, text_a=to_add, text_b=question, label="not_answerable"))
                uuid+=1
            samples = samples + contexts
            doc_lengths.append(len(contexts))
       
        preds = getPredictions(samples, model, tokenizer, max_seq, doc_lengths, args)
        
        tobe_sorted = []
        for pred, doc in zip(preds, docs):
            tobe_sorted.append((pred, doc[1]))
        
        tobe_sorted.sort(key= lambda x : x[0], reverse=True)
        return_documents.append(tobe_sorted[0:5])
    
    return zip(answers, return_documents, questions)



def get_score(answer_doc, match):
    """Search through all the top docs to see if they have the answer."""
    answer, docs, question = answer_doc
    for _, doc_id in docs:
        if has_answer(answer, doc_id, match):
            return 1
    return 0

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--doc-db', type=str, default=None,
                        help='Path to Document DB')
    parser.add_argument('--tokenizer', type=str, default='regexp')
    parser.add_argument('--n-docs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--match', type=str, default='string',
                        choices=['regex', 'string'])
    parser.add_argument('--classifier-path', type=str, default=None, help="path to classifier")
    parser.add_argument('--model-type', type=str, default=None)
    parser.add_argument('--max-seq', type=int, default=384)
    args = parser.parse_args()

    # start time
    start = time.time()

    # read all the data and store it
    logger.info('Reading data ...')
    questions = []
    answers = []

    for line in open(args.dataset):
        data = json.loads(line)
        question = data['question']
        answer = data['answer']
        questions.append(question)
        answers.append(answer)
   
    # get the closest docs for each question.
    logger.info('Initializing ranker...')
    ranker = retriever.get_class('tfidf')(tfidf_path=args.model)

    logger.info('Ranking...')
    closest_docs = ranker.batch_closest_docs(
        questions, k=args.n_docs, num_workers=args.num_workers
    )
    ranker = []

    tok_class = tokenizers.get_class(args.tokenizer)
    tok_opts = {}
    db_class = retriever.DocDB
    db_opts = {'db_path': args.doc_db}
    PROCESS_TOK = tok_class(**tok_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)

    answers_docs = rerankDocs(questions, answers, closest_docs, PROCESS_DB, args.classifier_path, args.max_seq, args)
    #answers_docs = zip(answers, closest_docs, questions)
    
    logger.info('Retrieving texts and computing scores...')
    # define processes
    tok_class = tokenizers.get_class(args.tokenizer)
    tok_opts = {}
    db_class = retriever.DocDB
    db_opts = {'db_path': args.doc_db}
    processes = ProcessPool(
        processes=args.num_workers,
        initializer=init,
        initargs=(tok_class, tok_opts, db_class, db_opts)
    )
    get_score_partial = partial(get_score, match=args.match)
    scores = processes.map(get_score_partial, answers_docs)

    filename = os.path.basename(args.dataset)
    stats = (
        "\n" + "-" * 50 + "\n" +
        "{filename}\n" +
        "Examples:\t\t\t{total}\n" +
        "Matches in top {k}:\t\t{m}\n" +
        "Match % in top {k}:\t\t{p:2.2f}\n" +
        "Total time:\t\t\t{t:2.4f} (s)\n"
    ).format(
        filename=filename,
        total=len(scores),
        k=args.n_docs,
        m=sum(scores),
        p=(sum(scores) / len(scores) * 100),
        t=time.time() - start,
    )
    print(stats)
