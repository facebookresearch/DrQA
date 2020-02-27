#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Full DrQA pipeline."""

import regex
import time
import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing import cpu_count
from multiprocessing.util import Finalize

from .. import tokenizers
from . import TRQA_DEFAULTS

from ..reader.transformer_reader import TransformerReader
from ..retriever.transformer_ranker import TransformerRanker

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Multiprocessing functions to fetch and tokenize text
# ------------------------------------------------------------------------------

PROCESS_TOK = None
PROCESS_DB = None
PROCESS_CANDS = None


def init(tokenizer_class, tokenizer_opts, db_class, db_opts, candidates=None):
    global PROCESS_TOK, PROCESS_DB, PROCESS_CANDS
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
    PROCESS_CANDS = candidates


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def tokenize_text(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)


# ------------------------------------------------------------------------------
# Main DrQA pipeline
# ------------------------------------------------------------------------------


class TrQA(object):
    # Target size for squashing short paragraphs together.
    # 0 = read every paragraph independently
    # infty = read all paragraphs together
    GROUP_LENGTH = 0

    def __init__(
            self,
            tokenizer=None,
            fixed_candidates=None,
            batch_size=128,
            cuda=True,
            num_workers=cpu_count(),
            db_config=None,
            ranker_config=None,
            ranker_model_type='roberta',
            ranker_model_name_or_path='',
            ranker_label_list=['0', '1'],
            ranker_output_mode='classification',
            per_gpu_eval_batch_size=2,
            reader_model_type='roberta',
            reader_model_name_or_path='',
            reader_lang_id=0,
            reader_n_best_size=20,
            reader_max_answer_length=1000,
    ):
        self.batch_size = batch_size
        self.fixed_candidates = fixed_candidates is not None
        self.cuda = cuda
        if not ranker_model_name_or_path:
            ranker_model_name_or_path = TRQA_DEFAULTS['transformer_ranker_model_name_or_path']
            ranker_model_type = TRQA_DEFAULTS['transformer_ranker_model_type']
        if not reader_model_name_or_path:
            reader_model_name_or_path = TRQA_DEFAULTS['transformer_reader_model_name_or_path']
            reader_model_type = TRQA_DEFAULTS['transformer_reader_model_type']

        logger.info('Initializing document ranker...')
        ranker_config = ranker_config or {}
        ranker_class = ranker_config.get('class', TRQA_DEFAULTS['ranker'])
        ranker_opts = ranker_config.get('options', {})
        self.ranker = ranker_class(**ranker_opts)

        if not tokenizer:
            self.tok_class = TRQA_DEFAULTS['tokenizer']
        else:
            self.tok_class = tokenizers.get_class(tokenizer)
        self.tok_opts = {'annotators': {}}

        # ElasticSearch is also used as backend if used as ranker
        if hasattr(self.ranker, 'es'):
            self.db_config = ranker_config
            self.db_class = ranker_class
            self.db_opts = ranker_opts
        else:
            self.db_config = db_config or {}
            self.db_class = db_config.get('class', TRQA_DEFAULTS['db'])
            self.db_opts = db_config.get('options', {})

        logger.info('Initializing tokenizers and document retrievers...')
        self.num_workers = num_workers

        self.transformer_ranker = TransformerRanker(model_type=ranker_model_type, model_name_or_path=ranker_model_name_or_path,
                                                    no_cuda=not cuda, label_list=ranker_label_list,
                                                    per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                                                    output_mode=ranker_output_mode)
        self.transformer_reader = TransformerReader(model_type=reader_model_type, model_name_or_path=reader_model_name_or_path,
                 lang_id=reader_lang_id, no_cuda=not cuda, per_gpu_eval_batch_size=per_gpu_eval_batch_size,
                 n_best_size=reader_n_best_size, max_answer_length=reader_max_answer_length,)

    def _split_doc(self, doc):
        """Given a doc, split it into chunks (by paragraph)."""
        curr = []
        curr_len = 0
        for split in regex.split(r'\n+', doc):
            split = split.strip()
            if len(split) == 0:
                continue
            # Maybe group paragraphs together until we hit a length limit
            if len(curr) > 0 and curr_len + len(split) > self.GROUP_LENGTH:
                yield ' '.join(curr)
                curr = []
                curr_len = 0
            curr.append(split)
            curr_len += len(split)
        if len(curr) > 0:
            yield ' '.join(curr)

    def process(self, query, candidates=None, top_n=1, n_docs=5,
                return_context=False):
        """Run a single query."""
        predictions = self.process_batch(
            [query], [candidates] if candidates else None,
            top_n, n_docs, return_context
        )
        return predictions[0]

    def process_batch(self, queries, candidates=None, top_n=1, n_docs=5,
                      return_context=False):
        """Run a batch of queries (more efficient)."""
        t0 = time.time()
        logger.info('Processing %d queries...' % len(queries))
        logger.info('Retrieving top %d docs...' % n_docs)

        # Rank documents for queries.
        if len(queries) == 1:
            ranked = [self.ranker.closest_docs(queries[0], k=n_docs)]
        else:
            ranked = self.ranker.batch_closest_docs(
                queries, k=n_docs, num_workers=self.num_workers
            )
        all_docids, all_doc_scores = zip(*ranked)

        # Flatten document ids and retrieve text from database.
        # We remove duplicates for processing efficiency.
        flat_docids = list({d for docids in all_docids for d in docids})
        did2didx = {did: didx for didx, did in enumerate(flat_docids)}

        with ProcessPool(
            self.num_workers,
            initializer=init,
            initargs=(self.tok_class, self.tok_opts, self.db_class, self.db_opts, self.fixed_candidates)
        ) as p:
            doc_texts = list(p.imap(fetch_text, flat_docids))

        # Split and flatten documents. Maintain a mapping from doc (index in
        # flat list) to split (index in flat list).
        flat_splits = []
        didx2sidx = []
        for text in doc_texts:
            splits = self._split_doc(text)
            didx2sidx.append([len(flat_splits), -1])
            for split in splits:
                flat_splits.append(split)
            didx2sidx[-1][1] = len(flat_splits)

        # Push through the tokenizers as fast as possible.

        # Group into structured example inputs. Examples' ids represent
        # mappings to their question, document, and split ids.
        examples = []
        for qidx in range(len(queries)):
            for rel_didx, did in enumerate(all_docids[qidx]):
                start, end = didx2sidx[did2didx[did]]
                for sidx in range(start, end):
                    if (len(queries[qidx].split()) > 0 and
                            len(flat_splits[sidx].split()) > 5):
                        examples.append({
                            'id': (qidx, rel_didx, sidx),
                            'question': queries[qidx],
                            'document': flat_splits[sidx],
                        })
        logger.info('Reading %d paragraphs...' % len(examples))
        questions_examples = {}
        for example in examples:
            qidx, rel_didx, sidx = example['id']
            if qidx not in questions_examples:
                questions_examples[qidx] = [example]
            else:
                questions_examples[qidx].append(example)
        questions_examples = [question_examples for question_examples in questions_examples.values()]
        # Push all examples through the document reader.
        ranked_examples = self.transformer_ranker.rank_questions_examples(questions_examples)
        qs_predictions = self.transformer_reader.answer_questions(ranked_examples)
        # Arrange final top prediction data
        all_predictions = []
        q_predictions = []
        for predictions in qs_predictions:
            for ids, nbests in predictions.items():
                best = nbests[0]
                qidx, rel_didx, sidx = ids.replace('(', '').replace(')', '').split(',')
                logger.info('qidx {}, rel_didx {} sidx {}'.format(qidx, rel_didx, sidx))
                qidx = int(qidx)
                rel_didx = int(rel_didx)
                sidx = int(sidx)
                start = flat_splits[sidx].find(best['text'])
                if start != -1 and best['text'] != '':
                    end = start + len(best['text'])
                else:
                    start = -1
                    end = -1
                prediction = {
                    'doc_id': all_docids[qidx][rel_didx],
                    'span': best['text'],
                    'doc_score': float(all_doc_scores[qidx][rel_didx]),
                    'span_score': best['probability'],
                    'context': {'text': flat_splits[sidx], 'start': start, 'end': end}
                }
                q_predictions.append(prediction)
            all_predictions.append(q_predictions)
        return all_predictions
