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
from pytorch_transformers import XLNetTokenizer
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

def recontruct_with_max_seq(paragraphs, tokenizer,  max_seq):
    ret = []
    to_add = ""
    for temp in paragraphs:
        len_temp = len(tokenizer.tokenize(temp))
        if len_temp > max_seq:
            if len(to_add) > 1:
                ret.append(to_add)
                to_add = ""
            ret.append(temp)
        elif len(tokenizer.tokenize(to_add)) + len_temp <= max_seq:
            to_add = to_add + temp
        else:
            ret.append(to_add)
            to_add = temp
    if len(to_add) > 1:
        ret.append(to_add)
    
    return ret

def split_and_check_hanswer(answer, doc_id, PROCESS_DB, PROCESS_TOK, tokenizer):
    text = PROCESS_DB.get_doc_text(doc_id)
    text = utils.normalize(text)
    paragraphs = text.strip('\n\n\n')
    paragraphs = paragraphs.split('\n\n')

    paragraphs = recontruct_with_max_seq(paragraphs, tokenizer, 384)
    
    has_answ = []
    for paragraph in paragraphs:
        has_answ.append(check_ans(answer, paragraph, PROCESS_TOK))

    return paragraphs, has_answ

def check_ans(answer, paragraph, tokenizer):

    text = tokenizer.tokenize(paragraph).words(uncased=True)
    for single_answer in answer:
        single_answer = utils.normalize(single_answer)
        single_answer = PROCESS_TOK.tokenize(single_answer)
        single_answer = single_answer.words(uncased=True)
            
        for i in range(0, len(text) - len(single_answer) + 1):
            if  single_answer == text[i: i + len(single_answer)]:
                return 1
    return 0


def get_has_answer(answer_doc, match, PROCESS_DB, PROCESS_TOK, tokenizer):
    """Search through all the top docs to see if they have the answer."""
    answer, doc_ids, _ = answer_doc
    doc_ids = doc_ids[0]
    ret = []
    res = []
    paras = []
    answs = []
    for i in range(len(doc_ids)):
        doc_id = doc_ids[i]
        para, answ = split_and_check_hanswer(answer, doc_id, PROCESS_DB, PROCESS_TOK, tokenizer)
        paras += para
        answs += answ
    if 1 in answs:
        indexes = np.where(np.array(answs) == 1)[0]
        print(indexes)
        positive = [paras[i] for i in indexes]
        negative = []
        neg_index = []
        max_nb_try = 3*len(paras)
        nb_try = 0
        print(max_nb_try)
        while len(negative) < len(positive) and nb_try < max_nb_try:
            nb_try += 1
            index = np.random.randint(len(paras))
            if index not in indexes and index not in neg_index:
                negative.append(paras[index])
                neg_index.append(index)
        ret += positive + negative
        res += [1 if i < len(positive) else 0 for i in range(2*len(positive))]
    return ret, res

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

    #answers_docs = rerankDocs(questions, answers, closest_docs, PROCESS_DB)
    answers_docs = zip(answers, closest_docs, questions)

    logger.info('Retrieving texts and computing scores...')
    has_answers = []

    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    amplified_Dataset = []
    for answer_doc in answers_docs:
        paras, answ = (get_has_answer(answer_doc, args.match, PROCESS_DB, PROCESS_TOK, tokenizer))
        if len(paras) > 0:
            amplified_Dataset.append({
                        "question" : answer_doc[2], 
                        "contexts" : paras, 
                        "answers" : answer_doc[0], 
                        "has_answer" : answ})

    print("saving dataset")
    print(len(amplified_Dataset))
    with open('random_balanced_amp_para_squad1.1_train.json', 'w') as fp:
        json.dump(amplified_Dataset, fp)
