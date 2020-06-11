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
import regex

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from drqa import retriever, tokenizers
from drqa.retriever import utils
from question_classifier.input_example import InputExample
from question_classifier import utils as bert_utils
from reader import Reader
from reranker import Reranker

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
        TensorDataset)
from tqdm import tqdm, trange
import logging
import pickle
from transformers import SquadExample

logger = logging.getLogger(__name__)


def reconstruct_with_max_seq(doc, max_seq, tokenizer):
    ret = []
    to_add = []
    len_to_add = 0
    for split in regex.split(r'\n+', doc) :
        split = split.strip()
        if len(split) == 0:
            continue
    
        len_split = len(tokenizer.tokenize(split))
        if len(to_add) > 0 and len_to_add + len_split > max_seq:
            ret.append(' '.join(to_add))
            to_add = []
            len_to_add = 0
        
        to_add.append(split)
        len_to_add += len_split

    if len(to_add) > 0:
        ret.append(' '.join(to_add))

    return ret

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

    parser.add_argument('--doc-db', type=str, default=None,
            help='Path to Document DB')
    parser.add_argument('--tokenizer', type=str, default='regexp')
    parser.add_argument('--n-docs', type=int, default=5)
    parser.add_argument('--tfidf', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=1)

    parser.add_argument('--max-seq', type=int, default=384)
    parser.add_argument('--reader-model-type', type=str, default=None)
    parser.add_argument('--reader-path', type=str, default=None)
    parser.add_argument('--reader-output-dir', type=str, default=None)
    
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--match', type=str, default='string',
            choices=['regex', 'string'])
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
    ranker = retriever.get_class('tfidf')(tfidf_path=args.tfidf)

    logger.info('Ranking...')
    closest_docs = ranker.batch_closest_docs(
            questions, k=args.n_docs, num_workers=args.num_workers
            )

    tok_class = tokenizers.get_class(args.tokenizer)
    tok_opts = {}
    db_class = retriever.DocDB
    db_opts = {'db_path': args.doc_db}
    PROCESS_TOK = tok_class(**tok_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)

    del ranker

    logger.info('Initializing Reader...')
    reader = Reader(args.reader_model_type, args.reader_path, args.reader_output_dir)
    reader.load_model()
    tokenizer = reader.get_tokenizer()

    logger.info('Splittind documents into passages...')
    documents = []
    docs_per_queston = []
    for doc_ids, _ in closest_docs:
        passages = []
        for doc_id in doc_ids:
            text = PROCESS_DB.get_doc_text(doc_id)
            passages = passages + reconstruct_with_max_seq(text, args.max_seq, tokenizer)
        docs_per_queston.append(len(passages))
        documents.append([p for p in passages])

    samples = []
    uuid = 0
    for question, docs in zip(questions, documents):
        for doc in docs:
            samples.append(InputExample(guid=uuid, text_a=question, text_b=doc, label='not_answerable'))
            uuid+=1

    
    preds =[0]*len(samples)

    
    preds_and_docs = []
    for index, pred in enumerate(preds):
        preds_and_docs.append((pred, samples[index].text_b))


    logger.info('preparing data for extraction...')
    squad_samples = []
    save = []
    begin = 0
    end = 0
    q_id=0
    for indice in docs_per_queston:
        end+=1
        document = preds_and_docs[begin*indice:end*indice]
        begin+=1
        
        c_id=0
        for doc in document:
            squad_samples.append(SquadExample(
                qas_id=str(q_id) + '_' + str(c_id),
                question_text=questions[q_id],
                context_text=doc[1],
                answer_text='',
                start_position_character=0,
                title=''))
            c_id+=1
        q_id+=1

    
    logger.info(squad_samples[0])
    logger.info('Evaluating...')
    reader.evaluate(squad_samples)

