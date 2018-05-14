import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable

def sort_batch(batch, lengths):
	#inputs: batch_size*_
	#lengths: batch_size
	sorted_lengths, sorted_indices = torch.sort(lengths, dim=0, descending=True)
	#sorted_lengths how to convert it to cuda?
	sorted_batch = batch[sorted_indices]
	original_indices = lengths.new(sorted_indices.shape[0]).zero_()
	for i,value in enumerate(sorted_indices):
		original_indices[value] = i 
	return sorted_batch, sorted_lengths, original_indices.long()

def unsort_batch(batch, original_indices):
	#inputs: batch_size*_	
	return batch[original_indices]

def sequence_mask(lengths):
	#lengths: batch_size
	batch_size = lengths.size(0)
	mask = lengths.new(batch_size, lengths.max()).zero_().long()
	for i in range(batch_size):
		mask[i][:lengths[i]] = 1
	return mask #batch_size*seq_len #Convert to a Variable?

def mask_sequence(sequence, mask):
	#sequence: batch_size*seq_len*_
	#mask: batch_size*seq_len
	mask = mask.float()
	mask = mask.unsqueeze(-1) #mask: batch_size*seq_len*1
	return sequence*mask #batch_size*seq_len*_

def masked_softmax(vector, mask): #change it to tensor and mask after the pytorch version accpeting dim
	#vector: flatten_size*num_words
	#mask: flatten_size*num_words
	result = functional.softmax(vector * mask)
	result = result * mask
	result = result / (result.sum(dim=1, keepdim=True) + 1e-13)
	return result #result: flatten_size*num_words

def attention_pooling(attention_weights, sequence):
	#sequence: batch_size*num_words*iembed
	#attention_weights: batch_size*num_words_new*num_words or batch_size*num_words
	if attention_weights.dim() == 2:
		pooled_sequence = torch.matmul(attention_weights.unsqueeze(1), sequence).squeeze(1) #pooled_sequence: batch_size*iembed
	elif attention_weights.dim() == 3:
		pooled_sequence = torch.matmul(attention_weights, sequence) #pooled_sequence: batch_size*num_words_new*iembed
	return pooled_sequence

def reverse_sequence(sequence, dim=1):
	#sequence: batch_size*num_words*iembed --> dim=1
	#mask: batch_size*num_words --> dim=1
	idx = [i for i in range(sequence.size(dim)-1, -1, -1)]
	#idx = Variable(torch.LongTensor(idx))
	idx = Variable(torch.LongTensor(idx)).cuda()
	reverse_sequence = sequence.index_select(dim, idx)
	return reverse_sequence

if __name__=='__main__':
	lengths = torch.LongTensor([4,2,3])
	#print(sequence_mask(lengths))
	#print(masked_softmax(Variable(torch.LongTensor([[1,2,3]]).float()),\
		#Variable(torch.LongTensor([[1,1,0]]).float())))
	sequence = Variable(torch.randn(3,5,4))
	mask = Variable(torch.LongTensor(3,5))
	print(sequence)
	print(reverse_sequence(sequence))
	print(mask)
	print(reverse_sequence(mask))
