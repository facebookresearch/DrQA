#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to convert the default SQuAD dataset to the format:

'{"question": "q1", "answer": ["a11", ..., "a1i"]}'
...
'{"question": "qN", "answer": ["aN1", ..., "aNi"]}'

"""

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

# Read dataset
with open(args.input) as f:
    dataset = json.load(f)

# Iterate and write question-answer pairs
with open(args.output, 'w') as f:
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                answer = [a['text'] for a in qa['answers']]
                f.write(json.dumps({'question': question, 'answer': answer}))
                f.write('\n')
