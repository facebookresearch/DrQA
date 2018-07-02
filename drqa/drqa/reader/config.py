#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Model architecture/optimization options for DrQA document reader."""

import argparse
import logging

logger = logging.getLogger(__name__)

# Index of arguments concerning the core model architecture
MODEL_ARCHITECTURE = {
    'model_type', 'embedding_dim', 'hidden_size', 'doc_layers',
    'question_layers', 'rnn_type', 'concat_rnn_layers', 'question_merge',
    'use_qemb', 'use_in_question', 'use_pos', 'use_ner', 'use_lemma', 'use_tf'
}

# Index of arguments concerning the model optimizer/training
MODEL_OPTIMIZER = {
    'fix_embeddings', 'optimizer', 'learning_rate', 'momentum', 'weight_decay',
    'rnn_padding', 'dropout_rnn', 'dropout_rnn_output', 'dropout_emb',
    'max_len', 'grad_clipping', 'tune_partial'
}


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_model_args(parser):
    parser.register('type', 'bool', str2bool)

    # Model architecture
    model = parser.add_argument_group('DrQA Reader Model Architecture')
    model.add_argument('--model-type', type=str, default='rnn',
                       help='Model architecture type')
    model.add_argument('--embedding-dim', type=int, default=300,
                       help='Embedding size if embedding_file is not given')
    model.add_argument('--hidden-size', type=int, default=128,
                       help='Hidden size of RNN units')
    model.add_argument('--doc-layers', type=int, default=3,
                       help='Number of encoding layers for document')
    model.add_argument('--question-layers', type=int, default=3,
                       help='Number of encoding layers for question')
    model.add_argument('--rnn-type', type=str, default='lstm',
                       help='RNN type: LSTM, GRU, or RNN')

    # Model specific details
    detail = parser.add_argument_group('DrQA Reader Model Details')
    detail.add_argument('--concat-rnn-layers', type='bool', default=True,
                        help='Combine hidden states from each encoding layer')
    detail.add_argument('--question-merge', type=str, default='self_attn',
                        help='The way of computing the question representation')
    detail.add_argument('--use-qemb', type='bool', default=True,
                        help='Whether to use weighted question embeddings')
    detail.add_argument('--use-in-question', type='bool', default=True,
                        help='Whether to use in_question_* features')
    detail.add_argument('--use-pos', type='bool', default=True,
                        help='Whether to use pos features')
    detail.add_argument('--use-ner', type='bool', default=True,
                        help='Whether to use ner features')
    detail.add_argument('--use-lemma', type='bool', default=True,
                        help='Whether to use lemma features')
    detail.add_argument('--use-tf', type='bool', default=True,
                        help='Whether to use term frequency features')

    # Optimization details
    optim = parser.add_argument_group('DrQA Reader Optimization')
    optim.add_argument('--dropout-emb', type=float, default=0.4,
                       help='Dropout rate for word embeddings')
    optim.add_argument('--dropout-rnn', type=float, default=0.4,
                       help='Dropout rate for RNN states')
    optim.add_argument('--dropout-rnn-output', type='bool', default=True,
                       help='Whether to dropout the RNN output')
    optim.add_argument('--optimizer', type=str, default='adamax',
                       help='Optimizer: sgd or adamax')
    optim.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate for SGD only')
    optim.add_argument('--grad-clipping', type=float, default=10,
                       help='Gradient clipping')
    optim.add_argument('--weight-decay', type=float, default=0,
                       help='Weight decay factor')
    optim.add_argument('--momentum', type=float, default=0,
                       help='Momentum factor')
    optim.add_argument('--fix-embeddings', type='bool', default=True,
                       help='Keep word embeddings fixed (use pretrained)')
    optim.add_argument('--tune-partial', type=int, default=0,
                       help='Backprop through only the top N question words')
    optim.add_argument('--rnn-padding', type='bool', default=False,
                       help='Explicitly account for padding in RNN encoding')
    optim.add_argument('--max-len', type=int, default=15,
                       help='The max span allowed during decoding')


def get_model_args(args):
    """Filter args for model ones.

    From a args Namespace, return a new Namespace with *only* the args specific
    to the model architecture or optimization. (i.e. the ones defined here.)
    """
    global MODEL_ARCHITECTURE, MODEL_OPTIMIZER
    required_args = MODEL_ARCHITECTURE | MODEL_OPTIMIZER
    arg_values = {k: v for k, v in vars(args).items() if k in required_args}
    return argparse.Namespace(**arg_values)


def override_model_args(old_args, new_args):
    """Set args to new parameters.

    Decide which model args to keep and which to override when resolving a set
    of saved args and new args.

    We keep the new optimation, but leave the model architecture alone.
    """
    global MODEL_OPTIMIZER
    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k]:
            if k in MODEL_OPTIMIZER:
                logger.info('Overriding saved %s: %s --> %s' %
                            (k, old_args[k], new_args[k]))
                old_args[k] = new_args[k]
            else:
                logger.info('Keeping saved %s: %s' % (k, old_args[k]))
    return argparse.Namespace(**old_args)
