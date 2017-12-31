# adapted from Head First Python, 2nd edition, by Paul Barry - chapter 10
# this is how a session variable allows you to pass values between pages 

from flask import Flask, session, request
from scripts.pipeline.interactive import process

# from func import addsev, openurl

app = Flask(__name__)

# def addnine(x):
# 	return x+9

@app.route('/json-example', methods=['POST']) #GET requests will be blocked
def json_example():
    req_data = request.get_json()

    output = processs(req_data['query'], req_data['dox'])


    # number = addsev(req_data['number'])
    # python_version = req_data['version_info']['python'] #two keys are needed because of the nested object
    # example = req_data['examples'][0] #an index is needed because of the array
    # boolean_test = req_data['boolean_test']

    return output

    '''
           The language value is: {}
           The framework value is: {}
           The Python version is: {}
           The item at index 0 in the example list is: {}
           # The boolean value is: {}'''.format(language, number, python_version, example, boolean_test)

if __name__ == '__main__':
    app.run(debug=True)


 # curl -H "Content-Type: application/json" -X POST -d '{"query" : "How much did you sell Twitch for?", "dox" : "https://molly.com/q?q=how%20should%20we%20decide%20which%20features%20to%20build?&id=7606"}' http://127.0.0.1:5000/json-example

