#  DrQA implementation

To run:

1. Create a virtual environment (olabot.1)
2. `pip install -r requirements.txt` 
3. Install pytorch. `'sudo pip install pytorch` should work. Else, you'll need to get from [here.](https://github.com/pytorch/pytorch)
4. Set CLASSPATH. `export CLASSPATH=$CLASSPATH:/path/to/corenlp/download/*` . If you don't do this, you're calls to corenlp tokenizer will Timeout. You can use spacy or others if you like, but I followed DrQA convention and stuck w/ corenlp.
5. `cd DrQA; python drqa_api.py`. This should set up a Flask endpoint on port `http://127.0.0.1:5000/`. 
6. To interact, send POST requests to this endpoint, as follows:

```curl -H "Content-Type: application/json" -X POST -d '{"query" : "Why go to Y Combinator?", "dox" : "https://molly.com/q?q=how%20should%20we%20decide%20which%20features%20to%20build?&id=7606"}' http://127.0.0.1:5000/json-example```

In this case, we crucially have a JSON input with a `query` and a url link to a set of JSON defined documents (`dox`). We'll need to do some massaging to make this more robust, but the pipeline works for this one JSON endpoint.  


#  Directory Contents

- DrQA. Key changes were made to `scripts/pipeline/interactive.py` and `drqa/pipeline/drqa.py`. 
- `file.json`. A clean representation of the 8 documents used to retreiver answers.
- `json_cleaning.ipynb`. An exploratory notebook for initially parsing JSON endpoint. 
- `requestments.txt` -- system state for this build. 