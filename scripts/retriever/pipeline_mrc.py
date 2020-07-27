#!/store/dyfar/venv/bin/python
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
import csv

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

    parser.add_argument('--reader-model-type', type=str, default=None)
    parser.add_argument('--reader-path', type=str, default=None)
    parser.add_argument('--reader-output-dir', type=str, default=None)
    
    parser.add_argument('--match', type=str, default='string',
            choices=['regex', 'string'])
    args = parser.parse_args()

    # start time
    start = time.time()

    # read all the data and store it
    logger.info('Reading data ...')
    questions = []
    answers = []

    with open(args.dataset, 'r', encoding='utf-8-sig') as f:
        data = list(csv.reader(f, delimiter='\t', quotechar=None))

    # get the closest docs for each question.

    logger.info('Reader ...')
    reader = Reader(args.reader_model_type, args.reader_path, args.reader_output_dir)
    reader.load_model()
    

    logger.info('creating samples ...')
    squad_samples = []
    for line in data:
        squad_samples.append(SquadExample(
            qas_id=line[0],
            question_text=line[1],
            context_text=line[2],
            answer_text='',
            start_position_character=0,
            title=''))


    reader.evaluate(squad_samples)

