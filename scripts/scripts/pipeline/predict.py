#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Run predictions using the full DrQA retriever-reader pipeline."""

import torch
import os
import time
import json
import argparse
import logging

from drqa import pipeline
from drqa.retriever import utils


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('--out-dir', type=str, default='/tmp',
                    help=("Directory to write prediction file to "
                          "(<dataset>-<model>-pipeline.preds)"))
parser.add_argument('--reader-model', type=str, default=None,
                    help="Path to trained Document Reader model")
parser.add_argument('--retriever-model', type=str, default=None,
                    help="Path to Document Retriever model (tfidf)")
parser.add_argument('--doc-db', type=str, default=None,
                    help='Path to Document DB')
parser.add_argument('--embedding-file', type=str, default=None,
                    help=("Expand dictionary to use all pretrained "
                          "embeddings in this file"))
parser.add_argument('--candidate-file', type=str, default=None,
                    help=("List of candidates to restrict predictions to, "
                          "one candidate per line"))
parser.add_argument('--n-docs', type=int, default=5,
                    help="Number of docs to retrieve per query")
parser.add_argument('--top-n', type=int, default=1,
                    help="Number of predictions to make per query")
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to use "
                          "(e.g. 'corenlp')"))
parser.add_argument('--no-cuda', action='store_true',
                    help="Use CPU only")
parser.add_argument('--gpu', type=int, default=-1,
                    help="Specify GPU device id to use")
parser.add_argument('--parallel', action='store_true',
                    help='Use data parallel (split across gpus)')
parser.add_argument('--num-workers', type=int, default=None,
                    help='Number of CPU processes (for tokenizing, etc)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Document paragraph batching size')
parser.add_argument('--predict-batch-size', type=int, default=1000,
                    help='Question batching size')
args = parser.parse_args()
t0 = time.time()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

if args.candidate_file:
    logger.info('Loading candidates from %s' % args.candidate_file)
    candidates = set()
    with open(args.candidate_file) as f:
        for line in f:
            line = utils.normalize(line.strip()).lower()
            candidates.add(line)
    logger.info('Loaded %d candidates.' % len(candidates))
else:
    candidates = None

logger.info('Initializing pipeline...')
DrQA = pipeline.DrQA(
    reader_model=args.reader_model,
    fixed_candidates=candidates,
    embedding_file=args.embedding_file,
    tokenizer=args.tokenizer,
    batch_size=args.batch_size,
    cuda=args.cuda,
    data_parallel=args.parallel,
    ranker_config={'options': {'tfidf_path': args.retriever_model,
                               'strict': False}},
    db_config={'options': {'db_path': args.doc_db}},
    num_workers=args.num_workers,
)


# ------------------------------------------------------------------------------
# Read in dataset and make predictions
# ------------------------------------------------------------------------------


logger.info('Loading queries from %s' % args.dataset)
queries = []
for line in open(args.dataset):
    data = json.loads(line)
    queries.append(data['question'])

model = os.path.splitext(os.path.basename(args.reader_model or 'default'))[0]
basename = os.path.splitext(os.path.basename(args.dataset))[0]
outfile = os.path.join(args.out_dir, basename + '-' + model + '-pipeline.preds')

logger.info('Writing results to %s' % outfile)
with open(outfile, 'w') as f:
    batches = [queries[i: i + args.predict_batch_size]
               for i in range(0, len(queries), args.predict_batch_size)]
    for i, batch in enumerate(batches):
        logger.info(
            '-' * 25 + ' Batch %d/%d ' % (i + 1, len(batches)) + '-' * 25
        )
        predictions = DrQA.process_batch(
            batch,
            n_docs=args.n_docs,
            top_n=args.top_n,
        )
        for p in predictions:
            f.write(json.dumps(p) + '\n')

logger.info('Total time: %.2f' % (time.time() - t0))
