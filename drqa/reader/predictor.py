#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""DrQA Document Reader predictor"""

import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

from .vector import vectorize, batchify
from .model import DocReader
from . import DEFAULTS, utils
from .. import tokenizers

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Tokenize + annotate
# ------------------------------------------------------------------------------

PROCESS_TOK = None


def init(tokenizer_class, annotators):
    global PROCESS_TOK
    PROCESS_TOK = tokenizer_class(annotators=annotators)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Predictor class.
# ------------------------------------------------------------------------------


class Predictor(object):
    """Load a pretrained DocReader model and predict inputs on the fly."""

    def __init__(self, model=None, tokenizer=None, normalize=True,
                 embedding_file=None, num_workers=None):
        """
        Args:
            model: path to saved model file.
            tokenizer: option string to select tokenizer class.
            normalize: squash output score to 0-1 probabilities with a softmax.
            embedding_file: if provided, will expand dictionary to use all
              available pretrained vectors in this file.
            num_workers: number of CPU processes to use to preprocess batches.
        """
        logger.info('Initializing model...')
        self.model = DocReader.load(model or DEFAULTS['model'],
                                    normalize=normalize)

        if embedding_file:
            logger.info('Expanding dictionary...')
            words = utils.index_embedding_words(embedding_file)
            added = self.model.expand_dictionary(words)
            self.model.load_embeddings(added, embedding_file)

        logger.info('Initializing tokenizer...')
        annotators = tokenizers.get_annotators_for_model(self.model)
        if not tokenizer:
            tokenizer_class = DEFAULTS['tokenizer']
        else:
            tokenizer_class = tokenizers.get_class(tokenizer)

        if num_workers is None or num_workers > 0:
            self.workers = ProcessPool(
                num_workers,
                initializer=init,
                initargs=(tokenizer_class, annotators),
            )
        else:
            self.workers = None
            self.tokenizer = tokenizer_class(annotators=annotators)

    def predict(self, document, question, candidates=None, top_n=1):
        """Predict a single document - question pair."""
        results = self.predict_batch([(document, question, candidates,)], top_n)
        return results[0]

    def predict_batch(self, batch, top_n=1):
        """Predict a batch of document - question pairs."""
        documents, questions, candidates = [], [], []
        for b in batch:
            documents.append(b[0])
            questions.append(b[1])
            candidates.append(b[2] if len(b) == 3 else None)
        candidates = candidates if any(candidates) else None

        # Tokenize the inputs, perhaps multi-processed.
        if self.workers:
            q_tokens = self.workers.map_async(tokenize, questions)
            d_tokens = self.workers.map_async(tokenize, documents)
            q_tokens = list(q_tokens.get())
            d_tokens = list(d_tokens.get())
        else:
            q_tokens = list(map(self.tokenizer.tokenize, questions))
            d_tokens = list(map(self.tokenizer.tokenize, documents))

        examples = []
        for i in range(len(questions)):
            examples.append({
                'id': i,
                'question': q_tokens[i].words(),
                'qlemma': q_tokens[i].lemmas(),
                'document': d_tokens[i].words(),
                'lemma': d_tokens[i].lemmas(),
                'pos': d_tokens[i].pos(),
                'ner': d_tokens[i].entities(),
            })

        # Stick document tokens in candidates for decoding
        if candidates:
            candidates = [{'input': d_tokens[i], 'cands': candidates[i]}
                          for i in range(len(candidates))]

        # Build the batch and run it through the model
        batch_exs = batchify([vectorize(e, self.model) for e in examples])
        s, e, score = self.model.predict(batch_exs, candidates, top_n)

        # Retrieve the predicted spans
        results = []
        for i in range(len(s)):
            predictions = []
            for j in range(len(s[i])):
                span = d_tokens[i].slice(s[i][j], e[i][j] + 1).untokenize()
                predictions.append((span, score[i][j].item()))
            results.append(predictions)
        return results

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
