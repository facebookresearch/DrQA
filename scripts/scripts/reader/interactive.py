#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to run the DrQA reader model interactively."""

import torch
import code
import argparse
import logging
import prettytable
import time

from drqa.reader import Predictor

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


# ------------------------------------------------------------------------------
# Commandline arguments & init
# ------------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None,
                    help='Path to model to use')
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to use "
                          "(e.g. 'corenlp')"))
parser.add_argument('--no-cuda', action='store_true',
                    help='Use CPU only')
parser.add_argument('--gpu', type=int, default=-1,
                    help='Specify GPU device id to use')
parser.add_argument('--no-normalize', action='store_true',
                    help='Do not softmax normalize output scores.')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

predictor = Predictor(args.model, args.tokenizer, num_workers=0,
                      normalize=not args.no_normalize)
if args.cuda:
    predictor.cuda()


# ------------------------------------------------------------------------------
# Drop in to interactive mode
# ------------------------------------------------------------------------------


def process(document, question, candidates=None, top_n=1):
    t0 = time.time()
    predictions = predictor.predict(document, question, candidates, top_n)
    table = prettytable.PrettyTable(['Rank', 'Span', 'Score'])
    for i, p in enumerate(predictions, 1):
        table.add_row([i, p[0], p[1]])
    print(table)
    print('Time: %.4f' % (time.time() - t0))


banner = """
DrQA Interactive Document Reader Module
>> process(document, question, candidates=None, top_n=1)
>> usage()
"""


def usage():
    print(banner)


code.interact(banner=banner, local=locals())
