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

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from drqa import retriever, tokenizers
from drqa.retriever import utils
from question_classifier.InputExample import InputExample

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


def has_answer(answer, doc_id, match, PROCESS_DB, PROCESS_TOK):
    """Check if a document contains an answer string.

    If `match` is string, token matching is done between the text and answer.
    If `match` is regex, we search the whole text with the regex.
    """
    text = PROCESS_DB.get_doc_text(doc_id)
    text = utils.normalize(text[:min(1000,len(text))])
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


def get_has_answer(answer_doc, match, PROCESS_DB, PROCESS_TOK):
    """Search through all the top docs to see if they have the answer."""
    answer, doc_ids, _ = answer_doc
    ret = []
    for i in range(len(doc_ids)):
        doc_id = doc_ids[i]
        if has_answer(answer, doc_id, match, PROCESS_DB, PROCESS_TOK):
            ret.append(i)
    return ret

def getPredictions(samples):
    return [1 for i in range(len(samples))]

def rerankDocs(questions, answers, closest_docs, db):

    documents = []
    for doc_ids, _ in closest_docs:
        batch = []
        for doc_id in doc_ids:
            text = db.get_doc_text(doc_id)
            batch.append((utils.normalize(text), doc_id))
        documents.append(batch)
    return_documents = []
    for question, docs in zip(questions, documents):
        samples = []
        for i in range(len(docs)):
            samples.append(InputExample(guid="%s" % i, text_a=docs[i][0], text_b=question))
        preds = getPredictions(samples)
        batch = []
        count = 0
        for i in range(len(preds)):
            if preds[i] == 1 and count < 5:
                batch.append(docs[i][1])
            elif count >= 5:
                break
        return_documents.append(batch)        
    return zip(answers, return_documents, questions)


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
    parser.add_argument('--n-docs', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=1)
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

    answers_docs = rerankDocs(questions, answers, closest_docs, PROCESS_DB)


    logger.info('Retrieving texts and computing scores...')
    has_answers = []

    amplified_Dataset = []
    for answer_doc in answers_docs:
        has_answers.append(get_has_answer(answer_doc, args.match, PROCESS_DB, PROCESS_TOK))
        if len(has_answers[-1]) > 0:
            for i in range(len(answer_doc[1])):
                text = PROCESS_DB.get_doc_text(answer_doc[1][i])
                answer_doc[1][i] = utils.normalize(text[:min(1000, len(text))])
            amplified_Dataset.append({
                "question" : answer_doc[2], 
                "contexts" : answer_doc[1], 
                "answers" : answer_doc[0], 
                "has_answer" : has_answers[-1]})

    print("saving dataset")
    print(len(amplified_Dataset))
    with open('amp_squad1.1_dev.json', 'w') as fp:
        json.dump(amplified_Dataset, fp)