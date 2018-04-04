#!/bin/bash

pushd data
mkdir embeddings
cd embeddings
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec &
wget http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip 
unzip glove.840B.300d.zip
popd
