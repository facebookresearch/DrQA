# Distant Supervision

## Generating Data

Datasets like SQuAD provide both supporting contexts and exact answer spans along with a question, but having this much supervision is rare. Indeed, the other QA datasets we consider in this project only contain question/answer pairs. Distant supervision is a way of automatically generating (noisy) full training examples (context, span, question) from these partial relations (QA pairs only) by using some heuristics.

`generate.py` runs a pipeline for generating distantly supervised datasets for DrQA. To do so, run:

```bash
python generate.py /path/to/dataset/dir dataset /path/to/output/dir
```

Optional arguments:

```
Dataset:
  --regex               Flag if answers are expressed as regexps
  --dev-split           Hold out for ds dev set (0.X)

Search Heuristic:
  --match-threshold     Minimum context overlap with question
  --char-max            Maximum allowed context length
  --char-min            Minimum allowed context length
  --window-sz           Use context on +/- window_sz for overlap measure

General:
  --max-ex              Maximum matches generated per question
  --n-docs              Number of docs retrieved per question
  --tokenizer           String option specifying tokenizer type to use (e.g. 'corenlp').
  --ranker              Ranking method for retrieving documents (e.g. 'tfidf')
  --db                  Database type (e.g. 'sqlite' for SqliteDB)
  --workers             Number of CPU processes (for tokenizing, etc).
```

The input dataset files must be in [format A](../../README.md#format-a).

The generated datasets are already in the preprocessed format required for the [Document Reader training](../reader/README.md#training). To combine different distantly supervised datasets, simply concatenate the files.

By default, the script will put all the generated data in the training set (specified as `.dstrain`). To hold out some data for fine tuning, adjust the `--dev-split` parameter (for multi-tasking the standard SQuAD dev set can be used). The generated dev set will end with `.dsdev`.

Note that if the dataset has answers in the form of regular expressions (e.g. CuratedTrec), the `--regex` flag must be set.

## Controlling Quality

Paragraphs are skipped if:

1. They are too long or short.
2. They don't contain a token match with the answer.
3. They don't contain token matches with named entities found in the question (using both NER recognizers from NLTK and the default for the `--tokenizer` option).
4. The overlap between the context and the question is too low.

Setting different thresholds for conditions 1 and 4 can change the quality of found matches. These are adjusted through the `--match-threshold`, `--char-max`, `--char-min`, and `--window-sz` parameters.

## Checking Results

To visualize the generated data, run:

```bash
python check_data.py /path/to/generated/file
```

This will allow you to manually iterate through the dataset and visually inspect examples. For example:

```bash
python scripts/distant/check_data.py data/generated/WikiMovies-train.dstrain
```

>Question: what films can be described by juzo itami ?
>
>Document: Itami 's debut as director was the movie \`\` Osōshiki ( _**The Funeral**_ ) '' in 1984 , at the age of 50 . This film proved popular in Japan and won many awards , including Japanese Academy Awards for Best Picture , Best Director , and Best Screenplay . However , it was his second movie , the \`\` noodle western '' \`\` Tampopo '' , that earned him international exposure and acclaim .

**Note:** The script in the repository is slightly modified from the original. Some of the logic was improved and it is faster. The number of DS instances it generates will be different from the paper, however. The performance is still similar, and the multitask models available in this repository were trained on data generated with the provided script.

If run with the default arguments, the expected numbers of generated examples are:

| Dataset       | Examples  |
| ------------- |:---------:|
| CuratedTrec   | 3670      |
| WebQuestions  | 7200      |
| WikiMovies    | 93927     |
| SQuAD         | 82690     |
