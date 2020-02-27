#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""DrQA reader utilities."""

import json
import time
import logging
import string
import regex as re

from collections import Counter
from .data import Dictionary

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------------------


def load_data(args, filename, skip_no_answer=False):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    """
    # Load JSON lines
    with open(filename) as f:
        examples = [json.loads(line) for line in f]

    # Make case insensitive?
    if args.uncased_question or args.uncased_doc:
        for ex in examples:
            if args.uncased_question:
                ex['question'] = [w.lower() for w in ex['question']]
            if args.uncased_doc:
                ex['document'] = [w.lower() for w in ex['document']]

    # Skip unparsed (start/end) examples
    if skip_no_answer:
        examples = [ex for ex in examples if len(ex['answers']) > 0]

    return examples


def load_text(filename):
    """Load the paragraphs only of a SQuAD dataset. Store as qid -> text."""
    # Load JSON file
    with open(filename) as f:
        examples = json.load(f)['data']

    texts = {}
    for article in examples:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                texts[qa['id']] = paragraph['context']
    return texts


def load_answers(filename):
    """Load the answers only of a SQuAD dataset. Store as qid -> [answers]."""
    # Load JSON file
    with open(filename) as f:
        examples = json.load(f)['data']

    ans = {}
    for article in examples:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                ans[qa['id']] = list(map(lambda x: x['text'], qa['answers']))
    return ans


# ------------------------------------------------------------------------------
# Dictionary building
# ------------------------------------------------------------------------------


def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file) as f:
        for line in f:
            w = Dictionary.normalize(line.rstrip().split(' ')[0])
            words.add(w)
    return words


def load_words(args, examples):
    """Iterate and index all the words in examples (documents + questions)."""
    def _insert(iterable):
        for w in iterable:
            w = Dictionary.normalize(w)
            if valid_words and w not in valid_words:
                continue
            words.add(w)

    if args.restrict_vocab and args.embedding_file:
        logger.info('Restricting to words in %s' % args.embedding_file)
        valid_words = index_embedding_words(args.embedding_file)
        logger.info('Num words in set = %d' % len(valid_words))
    else:
        valid_words = None

    words = set()
    for ex in examples:
        _insert(ex['question'])
        _insert(ex['document'])
    return words


def build_word_dict(args, examples):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Dictionary()
    for w in load_words(args, examples):
        word_dict.add(w)
    return word_dict


def top_question_words(args, examples, word_dict):
    """Count and return the most common question words in provided examples."""
    word_count = Counter()
    for ex in examples:
        for w in ex['question']:
            w = Dictionary.normalize(w)
            if w in word_dict:
                word_count.update([w])
    return word_count.most_common(args.tune_partial)


def build_feature_dict(args, examples):
    """Index features (one hot) from fields in examples and options."""
    def _insert(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

    feature_dict = {}

    # Exact match features
    if args.use_in_question:
        _insert('in_question')
        _insert('in_question_uncased')
        if args.use_lemma:
            _insert('in_question_lemma')

    # Part of speech tag features
    if args.use_pos:
        for ex in examples:
            for w in ex['pos']:
                _insert('pos=%s' % w)

    # Named entity tag features
    if args.use_ner:
        for ex in examples:
            for w in ex['ner']:
                _insert('ner=%s' % w)

    # Term frequency feature
    if args.use_tf:
        _insert('tf')
    return feature_dict


# ------------------------------------------------------------------------------
# Evaluation. Follows official evalutation script for v1.1 of the SQuAD dataset.
# ------------------------------------------------------------------------------


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Check if the prediction is a (soft) exact match with the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def regex_match_score(prediction, pattern):
    """Check if the prediction matches the given regular expression."""
    try:
        compiled = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE
        )
    except BaseException:
        logger.warn('Regular expression failed to compile: %s' % pattern)
        return False
    return compiled.match(prediction) is not None


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# ------------------------------------------------------------------------------
# Utility classes
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total
