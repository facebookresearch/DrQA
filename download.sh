#!/bin/bash
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -e

# Configure download location
DOWNLOAD_PATH="$DRQA_DATA"
if [ "$DRQA_DATA" == "" ]; then
    echo "DRQA_DATA not set; downloading to default path ('data')."
    DOWNLOAD_PATH="./data"
fi

# Get AWS hosted data
DOWNLOAD_PATH_TAR="$DOWNLOAD_PATH.tar.gz"

# Download main hosted data
wget -O "$DOWNLOAD_PATH_TAR" "https://dl.fbaipublicfiles.com/drqa/data.tar.gz"

# Untar
tar -xvf "$DOWNLOAD_PATH_TAR"

# Remove tar ball
rm "$DOWNLOAD_PATH_TAR"

# Get externally hosted data
DATASET_PATH="$DOWNLOAD_PATH/datasets"

# Get SQuAD train
wget -O "$DATASET_PATH/SQuAD-v1.1-train.json" "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
python scripts/convert/squad.py "$DATASET_PATH/SQuAD-v1.1-train.json" "$DATASET_PATH/SQuAD-v1.1-train.txt"

# Get SQuAD dev
wget -O "$DATASET_PATH/SQuAD-v1.1-dev.json" "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
python scripts/convert/squad.py "$DATASET_PATH/SQuAD-v1.1-dev.json" "$DATASET_PATH/SQuAD-v1.1-dev.txt"

# Download official eval for SQuAD
curl "https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/" >  "./scripts/reader/official_eval.py"

# Get WebQuestions train
wget -O "$DATASET_PATH/WebQuestions-train.json.bz2" "http://nlp.stanford.edu/static/software/sempre/release-emnlp2013/lib/data/webquestions/dataset_11/webquestions.examples.train.json.bz2"
bunzip2 -f "$DATASET_PATH/WebQuestions-train.json.bz2"
python scripts/convert/webquestions.py "$DATASET_PATH/WebQuestions-train.json" "$DATASET_PATH/WebQuestions-train.txt"
rm "$DATASET_PATH/WebQuestions-train.json"

# Get WebQuestions test
wget -O "$DATASET_PATH/WebQuestions-test.json.bz2" "http://nlp.stanford.edu/static/software/sempre/release-emnlp2013/lib/data/webquestions/dataset_11/webquestions.examples.test.json.bz2"
bunzip2 -f "$DATASET_PATH/WebQuestions-test.json.bz2"
python scripts/convert/webquestions.py "$DATASET_PATH/WebQuestions-test.json" "$DATASET_PATH/WebQuestions-test.txt"
rm "$DATASET_PATH/WebQuestions-test.json"

# Get freebase entities for WebQuestions
wget -O "$DATASET_PATH/freebase-entities.txt.gz" "https://dl.fbaipublicfiles.com/drqa/freebase-entities.txt.gz"
gzip -d "$DATASET_PATH/freebase-entities.txt.gz"

echo "DrQA download done!"
