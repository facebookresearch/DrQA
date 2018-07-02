# Document Retriever

## Storing the Documents

To efficiently store and access our documents, we store them in a sqlite database. The key is the `doc_id` and the value is the `text`.

To create a sqlite db from a corpus of documents, run:

```bash
python build_db.py /path/to/data /path/to/saved/db.db
```

Optional arguments:
```
--preprocess    File path to a python module that defines a `preprocess` function.
--num-workers   Number of CPU processes (for tokenizing, etc).
```

The data path can either be a path to a nested directory of files (such as what the [WikiExtractor](https://github.com/attardi/wikiextractor) script outputs) or a single file. Each file should consist of JSON-encoded documents that have `id` and `text` fields, one per line:

```python
{"id": "doc1", "text": "text of doc1"}
...
{"id": "docN", "text": "text of docN"}
```

`--preprocess /path/to/.py/file` is another optional argument that allows you to supply a python module that defines a `preprocess(doc_object)` function to filter/process documents before they are put in the db. See `prep_wikipedia.py` for an example.

## Building the TF-IDF N-grams

To build a TF-IDF weighted word-doc sparse matrix from the documents stored in the sqlite db, run:

```bash
python build_tfidf.py /path/to/doc/db /path/to/output/dir
```

Optional arguments:
```
--ngram         Use up to N-size n-grams (e.g. 2 = unigrams + bigrams). By default only ngrams without stopwords or punctuation are kept.
--hash-size     Number of buckets to use for hashing ngrams.
--tokenizer     String option specifying tokenizer type to use (e.g. 'corenlp').
--num-workers   Number of CPU processes (for tokenizing, etc).
```

The sparse matrix and its associated metadata will be saved to the output directory under `<db-name>-tfidf-ngram=<N>-hash=<N>-tokenizer=<T>.npz`.

## Interactive

The Document Retriever can also be used interactively (like the [full pipeline](../../README.md#quick-start-demo)).

```bash
python scripts/retriever/interactive.py --model /path/to/model
```

```
>>> process('question answering', k=5)

+------+-------------------------------+-----------+
| Rank |             Doc Id            | Doc Score |
+------+-------------------------------+-----------+
|  1   |       Question answering      |   327.89  |
|  2   |       Watson (computer)       |   217.26  |
|  3   |          Eric Nyberg          |   214.36  |
|  4   |   Social information seeking  |   212.63  |
|  5   | Language Computer Corporation |   184.64  |
+------+-------------------------------+-----------+
``` 