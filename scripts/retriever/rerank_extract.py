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
    parser.add_argument('--n-docs', type=int, default=200)
    parser.add_argument('--tfidf', type=str, default=None)
    parser.add_argument('--num-workers', type=int, default=1)

    parser.add_argument('--rerank-model-type', type=str, default=None)
    parser.add_argument('--rerank-path', type=str, default=None)
    parser.add_argument('--rerank-max-seq', type=int, default=384)
    parser.add_argument('--rerank-n-docs', type=int, default=20)

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

    questions = questions[0:1]
    answers = answers[0:1]
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

    ranker = []
    reranker = Reranker(args.rerank_model_type, args.rerank_path, args.rerank_max_seq)
    reranker.load_model()
    logger.info("reranking ...")

    documents = []
    docs_per_queston = []
    for doc_ids, _ in closest_docs:
        batch = []
        docs_per_queston.append(len(doc_ids))
        for doc_id in doc_ids:
            text = PROCESS_DB.get_doc_text(doc_id)
            batch.append((utils.normalize(text), doc_id))
        documents.append(batch)

    samples = []
    uuid = 0
    for question, docs in zip(questions, documents):
        for doc, doc_id in docs:
            samples.append(InputExample(guid=uuid, text_a=question, text_b=doc, label='not_answerable'))
            uuid+=1

    
    preds = reranker.evaluate(samples)
    reranker = []
    del reranker
    torch.cuda.empty_cache()

    logger.info('Reader ...')
    reader = Reader(args.reader_model_type, args.reader_path, args.reader_output_dir)
    reader.load_model()
    
    preds_and_docs = []
    for index, pred in enumerate(preds):
        preds_and_docs.append((pred, samples[index].text_a))

    squad_samples = []
    save = []
    begin = 0
    end = 0
    i = 0
    for indice in docs_per_queston:
        uuid=0
        end+=1
        to_sort = preds_and_docs[begin*indice:end*indice]
        begin+=1
        to_sort.sort(key= lambda x: x[0], reverse=True)
        save.append(to_sort)
        for doc in to_sort[0:min(args.rerank_n_docs, len(to_sort))]:
            squad_samples.append(SquadExample(
                qas_id=str(i) + '_' + str(uuid),
                question_text=questions[i],
                context_text=doc[1],
                answer_text='',
                start_position_character=0,
                title=''))
            uuid+=1
        i+=1


    with open(os.path.join(args.save_dir, "preds"), 'wb') as fp:
        pickle.dump(preds, fp)
    with open(os.path.join(args.save_dir, "reranked_samples"), 'wb') as fp:
        pickle.dump(save, fp)
    #reader.evaluate(squad_samples)

