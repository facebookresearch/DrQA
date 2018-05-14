import os 
import sys
import json

def compare(original, prediction1, prediction2, interactive=True):
	wtypedict1 = {'who' : [0,0], 'what': [0,0], 'where': [0,0], 'when': [0,0], 'how':[0,0], 'which':[0,0], 'why':[0,0]}
	wtypedict2 = {'who' : [0,0], 'what': [0,0], 'where': [0,0], 'when': [0,0], 'how':[0,0], 'which':[0,0], 'why':[0,0]}
	for line in original:
		line = line.strip()
		currentqa = json.loads(line)	
		id = currentqa['id']
		question = currentqa['question']
		document = currentqa['document']
		answers = currentqa['answers']
		print(' '.join(document))
		print(' ')
		print (' '.join(question))
		origanswers = []		
		for answer in answers:
			start = answer[0]
			end = answer[1]
			tokens = [document[i] for i in range(start, end+1)]
			print(' '.join(tokens))
			origanswers.append(' '.join(tokens))
		print('Prediction1: ', prediction1[id][0])
		print('Prediction2: ', prediction2[id][0])
		print(id)
		print('#############################')
		if interactive:
			input()
			continue
		for key in wtypedict1.keys():
			if key == question[0].lower():
				wtypedict1[key][1] += 1
				if prediction1[id][0][0] in origanswers:
					wtypedict1[key][0] += 1
				wtypedict2[key][1] += 1
				if prediction2[id][0][0] in origanswers:
					wtypedict2[key][0] += 1
	print(wtypedict1)
	print(wtypedict2)


if __name__=='__main__':
	originaljson = open(sys.argv[1], 'r')
	pred1json = json.load(open(sys.argv[2] , 'r'))
	pred2json = json.load(open(sys.argv[3], 'r'))
	interctive = sys.argv[4]
	if interctive == 'interactive':
		iact = True
	else:
		iact = False
	compare(originaljson, pred1json, pred2json, iact)
