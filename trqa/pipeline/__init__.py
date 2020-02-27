#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from ..tokenizers import SpacyTokenizer
from ..tokenizers import CoreNLPTokenizer
from ..retriever import TfidfDocRanker
from ..retriever import DocDB
from .. import DATA_DIR

TRQA_DEFAULTS = {
    'tokenizer': SpacyTokenizer,
    'ranker': TfidfDocRanker,
    'db': DocDB,
    'transformer_ranker_model_type': 'roberta',
    'transformer_ranker_model_name_or_path': os.path.join(DATA_DIR, 'transformer_models/roberta_ranker_model'),
    'transformer_reader_model_type': 'roberta',
    'transformer_reader_model_name_or_path': os.path.join(DATA_DIR, 'transformer_models/roberta_reader_model'),
}

DEFAULTS = {
    'tokenizer': CoreNLPTokenizer,
    'ranker': TfidfDocRanker,
    'db': DocDB,
    'reader_model': os.path.join(DATA_DIR, 'reader/multitask.mdl'),
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


from .drqa import DrQA


from .trqa import TrQA

def trqa_set_default(key, value):
    global TRQA_DEFAULTS
    TRQA_DEFAULTS[key] = value