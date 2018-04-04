# DrQA on Prince

## Initial Setup

Install anaconda python in $HOME/anaconda3

Download DrQA repo in $SCRATCH/DrQA and install DrQA
```bash
cd $SCRATCH
git clone https://github.com/jpatrickpark/DrQA.git
cd DrQA; pip install -r requirements.txt; python setup.py develop
```

Install CoreNLP in $SCRATCH/data/corenlp

```bash
./install_corenlp.sh
```

Download data

```bash
./download.sh
./download_glove.sh
```

Some additional directory creations

```bash
mkdir models
mkdir output
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
sbatch preprocess_dev.s
```

Train model
```bash
sbatch train.s
```

You can see the output of runs in output as well as models folder
