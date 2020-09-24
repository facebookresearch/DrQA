# Document Reader

## Preprocessing

`preprocess.py` takes a SQuAD-formatted dataset and outputs a preprocessed, training-ready file. Specifically, it handles tokenization, mapping character offsets to token offsets, and any additional featurization such as lemmatization, part-of-speech tagging, and named entity recognition.

To preprocess SQuAD (assuming both input and output files are in `data/datasets`):

```bash
python scripts/reader/preprocess.py data/datasets data/datasets --split SQuAD-v1.1-train
```
```bash
python scripts/reader/preprocess.py data/datasets data/datasets --split SQuAD-v1.1-dev
```
- _You need to have [SQuAD](../../README.md#qa-datasets) train-v1.1.json and dev-v1.1.json in data/datasets (here renamed as SQuAD-v1.1-<train/dev>.json)_

## Training

`train.py` is the main train script for the Document Reader.

To get started with training a model on SQuAD with our best hyper parameters:

```bash
python scripts/reader/train.py --embedding-file glove.840B.300d.txt --tune-partial 1000
```
- _You need to have the [glove embeddings](#note-on-word-embeddings) downloaded to data/embeddings/glove.840B.300d.txt._
- _You need to have done the preprocessing above._

The training has many options that you can tune:

```
Environment:
--no-cuda           Train on CPU, even if GPUs are available. (default: False)
--gpu               Run on a specific GPU (default: -1)
--data-workers      Number of subprocesses for data loading (default: 5)
--parallel          Use DataParallel on all available GPUs (default: False)
--random-seed       Random seed for all numpy/torch/cuda operations (for reproducibility).
--num-epochs        Train data iterations.
--batch-size        Batch size for training.
--test-batch-size   Batch size during validation/testing.

Filesystem:
--model-dir         Directory for saved models/checkpoints/logs (default: /tmp/drqa-models).
--model-name        Unique model identifier (.mdl, .txt, .checkpoint) (default: <generated uuid>).
--data-dir          Directory of training/validation data (default: data/datasets).
--train-file        Preprocessed train file (default: SQuAD-v1.1-train-processed-corenlp.txt).
--dev-file          Preprocessed dev file (default: SQuAD-v1.1-dev-processed-corenlp.txt).
--dev-json          Unprocessed dev file to run validation while training on (used to get original text for getting spans and answer texts) (default: SQuAD-v1.1-dev.json).
--embed-dir         Directory of pre-trained embedding files (default: data/embeddings).
--embedding-file    Space-separated pretrained embeddings file (default: None).

Saving/Loading:
--checkpoint        Save model + optimizer state after each epoch (default: False).
--pretrained        Path to a pretrained model to warm-start with (default: <empty>).
--expand-dictionary Expand dictionary of pretrained (--pretrained) model to include training/dev words of new data (default: False).

Preprocessing:
--uncased-question  Question words will be lower-cased (default: False).
--uncased-doc       Document words will be lower-cased (default: False).
--restrict-vocab    Only use pre-trained words in embedding_file (default: True).

General:
--official-eval     Validate with official SQuAD eval (default: True).
--valid-metric      The evaluation metric used for model selection (default: f1).
--display-iter      Log state after every <display_iter> epochs (default: 25).
--sort-by-len       Sort batches by length for speed (default: True).

DrQA Reader Model Architecture:
--model-type        Model architecture type (default: rnn).
--embedding-dim     Embedding size if embedding_file is not given (default: 300).
--hidden-size       Hidden size of RNN units (default: 128).
--doc-layers        Number of encoding layers for document (default: 3).
--question-layers   Number of encoding layers for question (default: 3).
--rnn-type          RNN type: LSTM, GRU, or RNN (default: lstm).

DrQA Reader Model Details:
--concat-rnn-layers Combine hidden states from each encoding layer (default: True).
--question-merge    The way of computing the question representation (default: self_attn).
--use-qemb          Whether to use weighted question embeddings (default: True).
--use-in-question   Whether to use in_question_* (cased, uncased, lemma) features (default: True).
--use-pos           Whether to use pos features (default: True).
--use-ner           Whether to use ner features (default: True).
--use-lemma         Whether to use lemma features (default: True).
--use-tf            Whether to use term frequency features (default: True).

DrQA Reader Optimization:
--dropout-emb           Dropout rate for word embeddings (default: 0.4).
--dropout-rnn           Dropout rate for RNN states (default: 0.4).
--dropout-rnn-output    Whether to dropout the RNN output (default: True).
--optimizer             Optimizer: sgd or adamax (default: adamax).
--learning-rate         Learning rate for SGD only (default: 0.1).
--grad-clipping         Gradient clipping (default: 10).
--weight-decay          Weight decay factor (default: 0).
--momentum              Momentum factor (default: 0).
--fix-embeddings        Keep word embeddings fixed (use pretrained) (default: True).
--tune-partial          Backprop through only the top N question words (default: 0).
--rnn-padding           Explicitly account for padding (and skip it) in RNN encoding (default: False).
--max-len MAX_LEN       The max span allowed during decoding (default: 15).
```

### Note on Word Embeddings

Using pre-trained word embeddings is very important for performance. The models we provide were trained with cased GloVe embeddings trained on Common Crawl, however we have also found that other embeddings such as FastText do quite well.

We suggest downloading the embeddings files and storing them under `data/embeddings/<file>.txt` (this is the default for `--embedding-dir`). The code expects space separated plain text files (\<token\> \<d1\> ... \<dN\>).

- [GloVe: Common Crawl (cased)](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip)
- [FastText: Wikipedia (uncased)](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)

## Predicting

`predict.py` uses a trained Document Reader model to make predictions for an input dataset.

Required arguments:
```
dataset               SQuAD-like dataset to evaluate on (format B).
```

Optional arguments:
```
--model             Path to model to use.
--embedding-file    Expand dictionary to use all pretrained embeddings in this file.
--out-dir           Directory to write prediction file to (<dataset>-<model>.preds).
--tokenizer         String option specifying tokenizer type to use (e.g. 'corenlp').
--num-workers       Number of CPU processes (for tokenizing, etc).
--no-cuda           Use CPU only.
--gpu               Specify GPU device id to use.
--batch-size        Example batching size (Reduce in case of OOM).
--top-n             Store top N predicted spans per example.
--official          Only store single top span instead of top N list. (The SQuAD eval script takes a dict of qid: span).
```

Note: The CoreNLP NER annotator is not fully deterministic (depends on the order examples are processed). Predictions may fluctuate very slightly between runs if `num-workers` > 1 and the model was trained with `use-ner` on.

Evaluation is done with the official_eval.py script from the SQuAD creators, available [here](https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/). It is also available by default at `scripts/reader/official_eval.py` after running `./download.sh`.

```bash
python scripts/reader/official_eval.py /path/to/format/B/dataset.json /path/to/predictions/with/--official/flag/set.json
```

## Interactive

The Document Reader can also be used interactively (like the [full pipeline](../../README.md#quick-start-demo)).

```bash
python scripts/reader/interactive.py --model /path/to/model
```

```
>>> text = "Mary had a little lamb, whose fleece was white as snow. And everywhere that Mary went the lamb was sure to go."
>>> question = "What color is Mary's lamb?"
>>> process(text, question)

+------+-------+---------+
| Rank |  Span |  Score  |
+------+-------+---------+
|  1   | white | 0.78002 |
+------+-------+---------+
```
