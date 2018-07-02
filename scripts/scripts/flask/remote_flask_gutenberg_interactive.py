import numpy as np
import flask
import io
import sys

import torch
import argparse
import code
import prettytable
import logging

from termcolor import colored
from drqa import pipeline
from drqa.retriever import utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

from gutenberg.query import get_etexts
from gutenberg.query import get_metadata

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--reader-model', type=str, default=None,
                    help='Path to trained Document Reader model')
parser.add_argument('--retriever-model', type=str, default=None,
                    help='Path to Document Retriever model (tfidf)')
parser.add_argument('--doc-db', type=str, default=None,
                    help='Path to Document DB')
parser.add_argument('--tokenizer', type=str, default=None,
                    help=("String option specifying tokenizer type to "
                          "use (e.g. 'corenlp')"))
parser.add_argument('--candidate-file', type=str, default=None,
                    help=("List of candidates to restrict predictions to, "
                          "one candidate per line"))
parser.add_argument('--no-cuda', action='store_true',
                    help="Use CPU only")
parser.add_argument('--gpu', type=int, default=-1,
                    help="Specify GPU device id to use")
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.cuda.set_device(args.gpu)
    logger.info('CUDA enabled (GPU %d)' % args.gpu)
else:
    logger.info('Running on CPU only.')

if args.candidate_file:
    logger.info('Loading candidates from %s' % args.candidate_file)
    candidates = set()
    with open(args.candidate_file) as f:
        for line in f:
            line = utils.normalize(line.strip()).lower()
            candidates.add(line)
    logger.info('Loaded %d candidates.' % len(candidates))
else:
    candidates = None

logger.info('Initializing pipeline...')
DrQA = pipeline.DrQA(
    cuda=args.cuda,
    fixed_candidates=candidates,
    reader_model=args.reader_model,
    ranker_config={'options': {'tfidf_path': args.retriever_model}},
    db_config={'options': {'db_path': args.doc_db}},
    tokenizer=args.tokenizer
)

# ------------------------------------------------------------------------------
# Drop in to interactive mode
# ------------------------------------------------------------------------------

def process(question, candidates=None, top_n=3, n_docs=3):
    torch.cuda.empty_cache()
    title = ''
    author = ''
    predictions = DrQA.process(
        question, candidates, top_n, n_docs, return_context=True
    )
    table = prettytable.PrettyTable(
        ['Rank', 'Answer', 'Doc-ID', 'Doc-Title', 'Doc-Author', 'Doc-Link', 'Answer Score', 'Doc Score']
    )
    for i, p in enumerate(predictions, 1):
        
        if not list(get_metadata('title', p['doc_id'])):
            title = 'Not Available'
        else:
            tittle = list(get_metadata('title', p['doc_id']))[0]

        if not list(get_metadata('author', p['doc_id'])):
            author = 'Not Available'
        else:
            author = list(get_metadata('author', p['doc_id']))[0]
       
        if not list(get_metadata('formaturi', p['doc_id'])):
            url = 'Not Available'
        else:
            url = list(get_metadata('formaturi', p['doc_id']))[0]

        table.add_row([i, p['span'], p['doc_id'], tittle, author, url, '%.5g' % p['span_score'], '%.5g' % p['doc_score']])
    print('Top Predictions:')
    print(table)
    strtable = table.get_string()
    print('\nContexts:')
    for p in predictions:
        text = p['context']['text']
        start = p['context']['start']
        end = p['context']['end']
        output = (text[:start] +
                  colored(text[start: end], 'green', attrs=['bold']) +
                  text[end:])
        print('[ Doc = %s ]' % p['doc_id'])
        print(output + '\n')
    retstring = strtable + '\n' + '[ Doc = ' + str(p['doc_id']) + ']' + '\n' + output + '\n'
    return retstring

# ## Ask Away!
@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False, "predictions": []}
    question = (flask.request.data).decode("utf-8")
    print("**********************************************************")
    print(question)
    y_output = process(question, candidates=None, top_n=1, n_docs=5)
    print(y_output)
    data["predictions"].append(str(y_output))
    
    #indicate that the request was a success
    data["success"] = True
    #return the data dictionary as a JSON response
    return flask.jsonify(data)




if __name__== "__main__":
    print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))

    app.run(host='0.0.0.0') # Ignore, Development server



