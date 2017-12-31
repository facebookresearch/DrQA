#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Full DrQA pipeline."""

import torch
import regex
import heapq
import math
import time
import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

from ..reader.vector import batchify
from ..reader.data import ReaderDataset, SortedBatchSampler
from .. import reader
from .. import tokenizers
from . import DEFAULTS

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
    # PROCESS_DB = db_class(**db_opts) # comment this when we are not using any Doc DB - just documents imported below
    # Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
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


class DrQA(object):
    # Target size for squashing short paragraphs together.
    # 0 = read every paragraph independently
    # infty = read all paragraphs together
    GROUP_LENGTH = 0

    def __init__(
            self,
            reader_model=None,
            embedding_file=None,
            tokenizer=None,
            fixed_candidates=None,
            batch_size=128,
            cuda=True,
            data_parallel=False,
            max_loaders=5,
            num_workers=None,
            db_config=None,
            ranker_config=None
    ):
        """Initialize the pipeline.

        Args:
            reader_model: model file from which to load the DocReader.
            embedding_file: if given, will expand DocReader dictionary to use
              all available pretrained embeddings.
            tokenizer: string option to specify tokenizer used on docs.
            fixed_candidates: if given, all predictions will be constrated to
              the set of candidates contained in the file. One entry per line.
            batch_size: batch size when processing paragraphs.
            cuda: whether to use the gpu.
            data_parallel: whether to use multile gpus.
            max_loaders: max number of async data loading workers when reading.
              (default is fine).
            num_workers: number of parallel CPU processes to use for tokenizing
              and post processing resuls.
            db_config: config for doc db.
            ranker_config: config for ranker.
        """
        self.batch_size = batch_size
        self.max_loaders = max_loaders
        self.fixed_candidates = fixed_candidates is not None
        self.cuda = cuda

        logger.info('Initializing document ranker...')
        ranker_config = ranker_config or {}
        ranker_class = ranker_config.get('class', DEFAULTS['ranker'])
        ranker_opts = ranker_config.get('options', {})
        # self.ranker = ranker_class(**ranker_opts) # comment out when we arent importing any npz model

        logger.info('Initializing document reader...')
        reader_model = reader_model or DEFAULTS['reader_model']
        self.reader = reader.DocReader.load(reader_model, normalize=False)
        if embedding_file:
            logger.info('Expanding dictionary...')
            words = reader.utils.index_embedding_words(embedding_file)
            added = self.reader.expand_dictionary(words)
            self.reader.load_embeddings(added, embedding_file)
        if cuda:
            self.reader.cuda()
        if data_parallel:
            self.reader.parallelize()

        if not tokenizer:
            tok_class = DEFAULTS['tokenizer']
        else:
            tok_class = tokenizers.get_class(tokenizer)
        annotators = tokenizers.get_annotators_for_model(self.reader)
        tok_opts = {'annotators': annotators}

        db_config = db_config or {}
        db_class = db_config.get('class', DEFAULTS['db'])
        db_opts = db_config.get('options', {})

        logger.info('Initializing tokenizers and document retrievers...')
        self.num_workers = num_workers
        self.processes = ProcessPool(
            num_workers,
            initializer=init,
            initargs=(tok_class, tok_opts, db_class, db_opts, fixed_candidates)
        )

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

    def _get_loader(self, data, num_loaders):
        """Return a pytorch data iterator for provided examples."""
        dataset = ReaderDataset(data, self.reader)
        sampler = SortedBatchSampler(
            dataset.lengths(),
            self.batch_size,
            shuffle=False
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_loaders,
            collate_fn=batchify,
            pin_memory=self.cuda,
        )
        return loader

    def process(self, query, dox, candidates=None, top_n=1, n_docs=5,
                return_context=False):
        """Run a single query."""
        predictions = self.process_batch(
            [query], dox, [candidates] if candidates else None,
            top_n, n_docs, return_context
        )
        return predictions[0]

    def process_batch(self, queries, dox, candidates=None, top_n=1, n_docs=5,
                      return_context=False):
        """Run a batch of queries (more efficient)."""
        t0 = time.time()
        logger.info('Processing %d queries...' % len(queries))
        logger.info('Retrieving top %d docs...' % n_docs)














        # Rank documents for queries.
        # if len(queries) == 1:
        #     ranked = [self.ranker.closest_docs(queries[0], k=n_docs)]
        # else:
        #     ranked = self.ranker.batch_closest_docs(
        #         queries, k=n_docs, num_workers=self.num_workers
        #     )
        # all_docids, all_doc_scores = zip(*ranked)
        # print(all_docids, all_doc_scores)

        import urllib.request, json 
        # dox = "https://molly.com/q?q=how%20should%20we%20decide%20which%20features%20to%20build?&id=7606"
        with urllib.request.urlopen(dox) as url:
            molly_data = json.loads(url.read().decode())


        molly_texts = []
        molly_ids = []
        for i, post in enumerate(molly_data['blog']):
            molly_ids.append(i)
            molly_texts.append(post.get('content'))





















        # Flatten document ids and retrieve text from database.
        # We remove duplicates for processing efficiency.
        # flat_docids = list({d for docids in all_docids for d in docids})
        # doc_texts = self.processes.map(fetch_text, flat_docids)
        # did2didx = {did: didx for didx, did in enumerate(flat_docids)}
        # print("DTEXTSSSSS")
        # print(doc_texts)
        # print(len(doc_texts))




        # Split and flatten documents. Maintain a mapping from doc (index in
        # flat list) to split (index in flat list).
        # flat_splits = []
        # didx2sidx = []
        # for text in doc_texts:
        #     splits = self._split_doc(text)
        #     didx2sidx.append([len(flat_splits), -1])
        #     print(didx2sidx)
        #     for split in splits:
        #         flat_splits.append(split)
        #     didx2sidx[-1][1] = len(flat_splits)
        #     print(didx2sidx)
        # print("FLAT SPLITS \n\n\n\n\n")
        # print(flat_splits[0])
        # print("didx2sidx {}".format(didx2sidx))


        # Push through the tokenizers as fast as possible.
        q_tokens = self.processes.map_async(tokenize_text, queries)
        # s_tokens = self.processes.map_async(tokenize_text, flat_splits)
        q_tokens = q_tokens.get()
        # s_tokens = s_tokens.get()



        molly_tokens = self.processes.map_async(tokenize_text, molly_texts)
        molly_tokens = molly_tokens.get()
        print('molly_tokens \n\n\n\n')
        print(molly_tokens)


        new_examples = []
        for i in range(len(molly_texts)):
            new_examples.append({
                            'id': (0, i, i),
                            'question': q_tokens[0].words(),
                            'qlemma': q_tokens[0].lemmas(),
                            'document': molly_tokens[i].words(),
                            'lemma': molly_tokens[i].lemmas(),
                            'pos': molly_tokens[i].pos(),
                            'ner': molly_tokens[i].entities(),
                        })
        print('new_examples \n\n\n\n')
        print(new_examples)



        # print('s_tokens {}'.format(s_tokens[0].words()))
        # Group into structured example inputs. Examples' ids represent
        # mappings to their question, document, and split ids.
        # print('all doc ids')
        # print(all_docids)
        # examples = []
        # for qidx in range(len(queries)):
        #     for rel_didx, did in enumerate(all_docids[qidx]):
        #         print(rel_didx)
        #         print(did)
        #         print(all_docids)
        #         print('didx2sidx {}'.format(didx2sidx))
        #         print('did2didx {}'.format(did2didx))
        #         start, end = didx2sidx[did2didx[did]]
        #         print('start {}'.format(start))
        #         print('end {}'.format(end))
        #         for sidx in range(start, end):
        #             if (len(q_tokens[qidx].words()) > 0 and
        #                     len(s_tokens[sidx].words()) > 0):
        #                 examples.append({
        #                     'id': (qidx, rel_didx, sidx),
        #                     'question': q_tokens[qidx].words(),
        #                     'qlemma': q_tokens[qidx].lemmas(),
        #                     'document': s_tokens[sidx].words(),
        #                     'lemma': s_tokens[sidx].lemmas(),
        #                     'pos': s_tokens[sidx].pos(),
        #                     'ner': s_tokens[sidx].entities(),
        #                 })
        # print('EXAMPLES \n\n\n\n\n')
        # # print(examples)
        # logger.info('Reading %d paragraphs...' % len(examples))

        











        # Push all examples through the document reader.
        # We decode argmax start/end indices asychronously on CPU.
        # result_handles = []
        # num_loaders = 0
        # # num_loaders = min(self.max_loaders, math.floor(len(examples) / 1e3))
        # print("num_loaders {}".format(num_loaders))
        # for batch in self._get_loader(examples, num_loaders):
        #     # if candidates or self.fixed_candidates:
        #     #     batch_cands = []
        #     #     for ex_id in batch[-1]:
        #     #         batch_cands.append({
        #     #             'input': s_tokens[ex_id[2]],
        #     #             'cands': candidates[ex_id[0]] if candidates else None
        #     # #         })
        #     #     handle = self.reader.predict(
        #     #         batch, batch_cands, async_pool=self.processes
        #     #     )
        #     # else:
        #     handle = self.reader.predict(batch, async_pool=self.processes)
        #     result_handles.append((handle, batch[-1], batch[0].size(0)))

        ################
        ################
        ################
        ################
        ################
        ################
        ################
        ################
        ################ TRY TO PUSH NEW EXAMPLES THROUGH PREDICTION? check! this is good
        ################
        ################
        ################
        ################
        ################
        ################
        ################
        ################
        ################


        # Push all examples through the document reader.
        # We decode argmax start/end indices asychronously on CPU.
        new_result_handles = []
        new_num_loaders = 0
        # num_loaders = min(self.max_loaders, math.floor(len(examples) / 1e3))
        print("num_loaders {}".format(new_num_loaders))
        for new_batch in self._get_loader(new_examples, new_num_loaders):
            # if candidates or self.fixed_candidates:
            #     batch_cands = []
            #     for ex_id in batch[-1]:
            #         batch_cands.append({
            #             'input': s_tokens[ex_id[2]],
            #             'cands': candidates[ex_id[0]] if candidates else None
            # #         })
            #     handle = self.reader.predict(
            #         batch, batch_cands, async_pool=self.processes
            #     )
            # else:
            new_handle = self.reader.predict(new_batch, async_pool=self.processes)
            new_result_handles.append((new_handle, new_batch[-1], new_batch[0].size(0)))














        # Iterate through the predictions, and maintain priority queues for
        # top scored answers for each question in the batch.
        # queues = [[] for _ in range(len(queries))]
        # for result, ex_ids, batch_size in result_handles:
        #     s, e, score = result.get()
        #     for i in range(batch_size):
        #         # We take the top prediction per split.
        #         if len(score[i]) > 0:
        #             item = (score[i][0], ex_ids[i], s[i][0], e[i][0])
        #             queue = queues[ex_ids[i][0]]
        #             if len(queue) < top_n:
        #                 heapq.heappush(queue, item)
        #             else:
        #                 heapq.heappushpop(queue, item)






         ################
        ################
        ################
        ################
        ################
        ################
        ################
        ################
        ################ TRY TO PUSH new_result_handles. check! i think so.
        ################
        ################
        ################
        ################
        ################
        ################
        ################
        ################
        ################

                # Iterate through the predictions, and maintain priority queues for
        # top scored answers for each question in the batch.
        new_queues = [[] for _ in range(len(queries))]
        for new_result, new_ex_ids, new_batch_size in new_result_handles:
            new_s, new_e, new_score = new_result.get()
            for i in range(new_batch_size):
                # We take the top prediction per split.
                if len(new_score[i]) > 0:
                    item = (new_score[i][0], new_ex_ids[i], new_s[i][0], new_e[i][0])
                    queue = new_queues[new_ex_ids[i][0]]
                    if len(queue) < top_n:
                        heapq.heappush(queue, item)
                    else:
                        heapq.heappushpop(queue, item)


        # Arrange final top prediction data.
        # all_predictions = []
        # for queue in queues:
        #     predictions = []
        #     while len(queue) > 0:
        #         score, (qidx, rel_didx, sidx), s, e = heapq.heappop(queue)
        #         prediction = {
        #             'doc_id': all_docids[qidx][rel_didx],
        #             'span': s_tokens[sidx].slice(s, e + 1).untokenize(),
        #             'doc_score': float(all_doc_scores[qidx][rel_didx]),
        #             'span_score': float(score),
        #         }
        #         if return_context:
        #             prediction['context'] = {
        #                 'text': s_tokens[sidx].untokenize(),
        #                 'start': s_tokens[sidx].offsets()[s][0],
        #                 'end': s_tokens[sidx].offsets()[e][1],
        #             }
        #         predictions.append(prediction)
        #     all_predictions.append(predictions[-1::-1])

        # logger.info('Processed %d queries in %.4f (s)' %
        #             (len(queries), time.time() - t0))





        ### try last step?

        new_all_predictions = []
        for queue in new_queues:
            new_predictions = []
            while len(queue) > 0:
                new_score, (new_qidx, new_rel_didx, new_sidx), new_s, new_e = heapq.heappop(queue)
                new_prediction = {
                    'doc_id': molly_ids[new_qidx], #[new_rel_didx],
                    'span': molly_tokens[new_sidx].slice(new_s, new_e + 1).untokenize(),
                    # 'doc_score': float(new_all_doc_scores[qidx][rel_didx]),
                    'span_score': float(new_score),
                }
                if return_context:
                    new_prediction['context'] = {
                        'text': molly_tokens[new_sidx].untokenize(),
                        'start': molly_tokens[new_sidx].offsets()[new_s][0],
                        'end': molly_tokens[new_sidx].offsets()[new_e][1],
                    }
                new_predictions.append(new_prediction)
            new_all_predictions.append(new_predictions[-1::-1])

        logger.info('Processed %d queries in %.4f (s)' %
                    (len(queries), time.time() - t0))







        print("new all predictions \n\n\n")
        print(new_all_predictions[0]) # This is the JSON I wanna return.

        return new_all_predictions



