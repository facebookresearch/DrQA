#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to visually inspect generated data."""

import argparse
import json
from termcolor import colored

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
args = parser.parse_args()

with open(args.file) as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        question = ' '.join(data['question'])
        start, end = data['answers'][0]
        doc = data['document']
        pre = ' '.join(doc[:start])
        ans = colored(' '.join(doc[start: end + 1]), 'red', attrs=['bold'])
        post = ' '.join(doc[end + 1:])
        print('-' * 50)
        print('Question: %s' % question)
        print('')
        print('Document: %s' % (' '.join([pre, ans, post])))
        input()
