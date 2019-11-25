#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to read in and store documents in a sqlite database."""

import argparse
import sqlite3
import json
import os
import logging
import importlib.util
import uuid

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
from drqa.retriever import utils
from transformers import BertTokenizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------
PREPROCESS_FN = None
TOKENIZER = None
MAX_SEQ = 384
def init(filename):
    global PREPROCESS_FN
    if filename:    
        PREPROCESS_FN = import_module(filename).preprocess


def import_module(filename):
    """Import a module given a full path to the file."""
    spec = importlib.util.spec_from_file_location('doc_filter', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)

def reconstruct_with_max_seq(docs, max_seq):
    ret = []
    to_add = ""
    for temp in docs:
        len_temp = len(TOKENIZER.tokenize(temp))
        if len_temp > max_seq:
            if len(to_add) > 1:
                ret.append((str(uuid.uuid4()), to_add))
                to_add = ""
            ret.append((str(uuid.uuid4()), temp))
        elif len(TOKENIZER.tokenize(to_add)) + len_temp <= max_seq:
            to_add = to_add + temp 
        else:
            ret.append((str(uuid.uuid4()), to_add))
            to_add = temp
    if len(to_add) > 1:
        ret.append((str(uuid.uuid4()), to_add))

    return ret



def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    global PREPROCESS_FN
    documents = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Maybe preprocess the document with custom function
            if PREPROCESS_FN:
                doc = PREPROCESS_FN(doc)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            doc['text'] = doc['text'].strip("\n\n\n")
            docs = doc['text'].split("\n\n")
            docs = reconstruct_with_max_seq(docs, MAX_SEQ)
            documents += docs
    return documents

def store_contents(data_path, save_path, preprocess, num_workers=None):
    """Preprocess and store a corpus of documents in sqlite.

    #Args:
    #    data_path: Root path to directory (or directory of directories) of files
    #      containing json encoded documents (must have `id` and `text` fields).
    #    save_path: Path to output sqlite db.
    #    preprocess: Path to file defining a custom `preprocess` function. Takes
    #      in and outputs a structured doc.
    #    num_workers: Number of parallel processes to use when reading docs.
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text);")

    workers = ProcessPool(num_workers, initializer=init, initargs=(preprocess,))
    files = [f for f in iter_files(data_path)]
    count = 0
    with tqdm(total=len(files)) as pbar:
        for pairs in tqdm(workers.imap_unordered(get_contents, files)):
            count += len(pairs)
            c.executemany("INSERT INTO documents VALUES (?,?)", pairs)
            pbar.update()
    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='/path/to/data')
    parser.add_argument('save_path', type=str, help='/path/to/saved/db.db')
    parser.add_argument('--preprocess', type=str, default=None,
                        help=('File path to a python module that defines '
                              'a `preprocess` function'))
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    parser.add_argument('--max-seq', type=int, default=384)
    args = parser.parse_args()

    MAX_SEQ = args.max_seq
    TOKENIZER = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking-finetuned-squad')
    print(MAX_SEQ)
    store_contents(
        args.data_path, args.save_path, args.preprocess, args.num_workers)
