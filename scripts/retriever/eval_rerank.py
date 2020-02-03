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


def get_score(answer_doc, match):
    """Search through all the top docs to see if they have the answer."""
    answer, doc_ids = answer_doc
    for doc_id in doc_ids:
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
    ids = []
    docs_per_queston = []
    for doc_ids, _ in closest_docs:
        batch = []
        docs_per_queston.append(len(doc_ids))
        for doc_id in doc_ids:
            text = PROCESS_DB.get_doc_text(doc_id)
            batch.append((utils.normalize(text), doc_id))
            ids.append(doc_id)
        documents.append(batch)

    samples = []
    uuid = 0
    for question, docs in zip(questions, documents):
        for doc, doc_id in docs:
            samples.append(InputExample(guid=uuid, text_a=doc, text_b=question, label='not_answerable'))
            uuid+=1

    
    preds = reranker.evaluate(samples)
    reranker = []
    del reranker
    torch.cuda.empty_cache()

    preds_and_docs = []
    for pred, doc_id in zip(preds, ids):
        preds_and_docs.append((pred, doc_id))

    save = []
    begin = 0
    end = 0
    i = 0
    reranked_docs = []
    for indice in docs_per_queston:
        end+=1
        to_sort = preds_and_docs[begin*indice:end*indice]
        begin+=1
        to_sort.sort(key= lambda x: x[0], reverse=True)
        save.append(to_sort)
        batch = []
        for doc in to_sort[0:min(args.rerank_n_docs, len(to_sort))]:
            batch.append(doc[1]),
        reranked_docs.append(batch)



    answers_docs = zip(answers, reranked_docs)

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

    # compute the scores for each pair, and print the statistics
    logger.info('Retrieving and computing scores...')
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
        k=args.rerank_n_docs,
        m=sum(scores),
        p=(sum(scores) / len(scores) * 100),
        t=time.time() - start,
    )

    print(stats)
