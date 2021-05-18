#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
import sys

with open('README.md', encoding='utf8') as f:
    readme = f.read()

with open('LICENSE', encoding='utf8') as f:
    license = f.read()

with open('requirements.txt', encoding='utf8') as f:
    reqs = f.read()

setup(
    name='drqa',
    version='0.1.0',
    description='Reading Wikipedia to Answer Open-Domain Questions',
    long_description=readme,
    license=license,
    python_requires='>=3.5',
    packages=find_packages(exclude=('data')),
    install_requires=reqs.strip().split('\n'),
)
