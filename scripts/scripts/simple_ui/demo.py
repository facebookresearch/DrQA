import numpy as np
import flask
import io
import sys
import pandas as pd
import csv

import torch
import argparse
import code
import prettytable
import logging
from colorama import init
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

import bottle
from bottle import route, run
import threading

from time import sleep

app = bottle.Bottle()
query = []
response = ""

parser = argparse.ArgumentParser()
parser.add_argument('--reader-model', type=str, default=None,
                    help='Path to trained Document Reader model')
parser.add_argument('--retriever-model', type=str, default='/data/MRC_Google_Compete/Gutenberg/gutenberg_children_TF_IDF/gutenbergchildrendb-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz',
                    help='Path to Document Retriever model (tfidf)')
parser.add_argument('--doc-db', type=str, default='/data/MRC_Google_Compete/Gutenberg/gutenberg_children_db/gutenbergchildrendb.db',
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

def process(question, candidates=None, top_n=3, n_docs=3):
    torch.cuda.empty_cache()
    init()
    title = ''
    author = ''
    predictions = DrQA.process(
        question, candidates, top_n, n_docs, return_context=True
    )
    ptable = prettytable.PrettyTable(
        ['Rank', 'Answer', 'Doc-Title', 'Doc-Author']
    )
    for i, p in enumerate(predictions):
        
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

        ptable.add_row([i+1, p['span'], tittle, author])

 
    with open('/data/MRC_Google_Compete/DrQA/result/output.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Answer', 'Doc-Title', 'Doc-Author'])
        for i, p in enumerate(predictions):      
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
            writer.writerow([str(i+1), str(p['span']), tittle, author])
    
    df = pd.read_csv("/data/MRC_Google_Compete/DrQA/result/output.csv")
    strtable = df.to_html()
    return strtable
    

   

@app.get("/")
def home():
    with open('/data/MRC_Google_Compete/DrQA/scripts/simple_ui/demo.html', 'r') as fl:
        html = fl.read()
        return html

@app.post('/answer')
def answer():
    question = bottle.request.json['question']
    global query, response
    query = (question)
    while not response:
        sleep(0.1)
    #print("received response: {}".format(response))
    Final_response = {"answer": response}
    response = []
    return Final_response

class Demo():
    def __init__(self):
        run_event = threading.Event()
        run_event.set()
        threading.Thread(target=self.demo_backend, args = [run_event]).start()
        app.run(port=5000, host='0.0.0.0')
        try:
            while 1:
                sleep(.1)
        except KeyboardInterrupt:
            print("Closing server...")
            run_event.clear()

    def demo_backend(self, run_event):
        global query, response
        while run_event.is_set():
            sleep(0.1)
            if query:
                response = process(query, candidates=None, top_n=3, n_docs=3)
                #print(response)
                query = ""