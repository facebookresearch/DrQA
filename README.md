# DrQA on Prince

## Initial Setup

Install anaconda python in $HOME/anaconda3

Download DrQA repo in $SCRATCH/DrQA

Install CoreNLP in $SCRATCH/data/corenlp

```bash
./install_corenlp.sh
```

Download data

```bash
./download.sh
```

Some additional directory creations

```bash
mkdir $SCRATCH/models
mkdir $SCRATCH/DrQA/output
```

Test corenlp
```python
from drqa.tokenizers import CoreNLPTokenizer
tok = CoreNLPTokenizer()
tok.tokenize('hello world').words()  # Should complete immediately
```

Test run (will run if corenlp is installed correctly)
```bash
sbatch interactive.s
```

Create preprocessed input data
```bash
sbatch preprocess.s
```

Train model
```bash
sbatch preprocess.s
```
