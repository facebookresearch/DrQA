import torch
import torch.nn as nn
from torch.autograd import Variable

from . import utils1
from . import similarity

class UniDirAttention(nn.Module):

	def __init__(self, dict_args):
		super(UniDirAttention, self).__init__()
		self.similarity_function_name = dict_args['similarity_function']
		#dict_args should contain the arguments for the similarity function as well
		self.similarity_function = None
		if self.similarity_function_name == 'DotProduct':
			self.similarity_function = similarity.DotProductSimilarity(dict_args)
		elif self.similarity_function_name == 'WeightedInputsConcatenation':
			self.similarity_function = similarity.LinearConcatenationSimilarity(dict_args)
		elif self.similarity_function_name == 'WeightedInputsDotConcatenation':
			self.similarity_function = similarity.LinearConcatenationDotSimilarity(dict_args)
		elif self.similarity_function_name == 'WeightedSumProjection':
			self.similarity_function = similarity.LinearProjectionSimilarity(dict_args)
		elif self.similarity_function_name == 'ProjectionSimilaritySharedWeights':
			self.similarity_function = dict_args['similarity_function_pointer']

	def forward(self, sequence, vector, sequence_mask, vector2=None, softmax=True):
		#vector: batch_size*iembed1
		#sequence: batch_size*num_words*iembed2
		if self.similarity_function_name == 'DotProduct':
			assert (sequence.size(2) == vector.size(1)),"iembed1 and iembed2 should be same for dotproduct similarity" 
		vector_tiled = vector.unsqueeze(1).expand(vector.size(0), sequence.size(1), vector.size(1)) #vector_tiled: batch_size*num_words*iembed1

		if vector2 is None:
			print(sequence.size())
			print(vector_tiled.size())
			similarity_vector = self.similarity_function(sequence, vector_tiled) #similarity_vector: batch_size*num_words
		else:
			#To be used only with 'ProjectionSimilaritySharedWeights' else concatenate vectors to form a single vector
			vector2_tiled = vector2.unsqueeze(1).expand(vector2.size(0), sequence.size(1), vector2.size(1)) #vector2_tiled: batch_size*num_words*iembed3
			similarity_vector = self.similarity_function(sequence, vector_tiled, vector2_tiled) #similarity_vector: batch_size*num_words

		sequence_attention_weights = utils1.masked_softmax(similarity_vector, sequence_mask.float())

		vector_sequence_attention = utils1.attention_pooling(sequence_attention_weights, sequence) #vector_sequence_attention: batch_size*iembed2

		if softmax:
			return vector_sequence_attention, sequence_attention_weights
		else:
			return vector_sequence_attention, similarity_vector			



class UniDirAttentionItr(nn.Module):

	def __init__(self, dict_args):
		super(UniDirAttentionItr, self).__init__()
		self.similarity_function_name = dict_args['similarity_function']
		#dict_args should contain the arguments for the similarity function as well
		self.similarity_function = None
		if self.similarity_function_name == 'DotProduct':
			self.similarity_function = similarity.DotProductSimilarity(dict_args)
		elif self.similarity_function_name == 'WeightedInputsConcatenation':
			self.similarity_function = similarity.LinearConcatenationSimilarity(dict_args)
		elif self.similarity_function_name == 'WeightedInputsDotConcatenation':
			self.similarity_function = similarity.LinearConcatenationDotSimilarity(dict_args)
		elif self.similarity_function_name == 'WeightedSumProjection':
			self.similarity_function = similarity.LinearProjectionSimilarity(dict_args)
		elif self.similarity_function_name == 'ProjectionSimilaritySharedWeights':
			self.similarity_function = dict_args['similarity_function_pointer']

	def forward(self, sequence, vector, sequence_mask, vector2=None, softmax=True):
		#vector: batch_size*iembed1
		#sequence: batch_size*num_words*iembed2
		if self.similarity_function_name == 'DotProduct':
			assert (sequence.size(2) == vector.size(1)),"iembed1 and iembed2 should be same for dotproduct similarity" 
		vector_tiled = vector.unsqueeze(1) #vector_tiled: batch_size*1*iembed1
		vector2_tiled = None
		if vector2 is not None:
			#To be used only with 'ProjectionSimilaritySharedWeights' else concatenate vectors to form a single vector
			vector2_tiled = vector2.unsqueeze(1) #vector2_tiled: batch_size*1*iembed3

		sequence = sequence.permute(1,0,2) #sequence: num_words*batch_size*iembed2
		num_words, batch_size, input_dim = sequence.size()
		similarity_vector = Variable(sequence.data.new(num_words,batch_size).zero_()) #similarity_vector: num_words*batch_size

		for ith_item in range(num_words):
			sequence_vector = sequence[ith_item] #sequence_vector: batch_size*iembed2
			sequence_vector = sequence_vector.unsqueeze(1) #sequence_vector: batch_size*1*iembed2
			if vector2 is not None:
				similarity_vector_itr = self.similarity_function(sequence_vector, vector_tiled, vector2_tiled)
			else:
				similarity_vector_itr = self.similarity_function(sequence_vector, vector_tiled)				
			similarity_vector[ith_item] = similarity_vector_itr.squeeze()
		similarity_vector = similarity_vector.permute(1,0) #similarity_vector: batch_size*num_words
		sequence = sequence.permute(1,0,2) #sequence: batch_size*num_words*iembed2

		sequence_attention_weights = utils.masked_softmax(similarity_vector, sequence_mask.float())

		vector_sequence_attention = utils.attention_pooling(sequence_attention_weights, sequence) #vector_sequence_attention: batch_size*iembed2

		#Will it save some memory?
		if self.training:
			sequence_attention_weights = None

		if softmax:
			return vector_sequence_attention, sequence_attention_weights
		else:
			return vector_sequence_attention, similarity_vector			


if __name__=='__main__':
	unidir = UniDirAttention({'similarity_function': 'WeightedSumProjection', 'sequence1_dim':6, 'sequence2_dim':20, 'projection_dim':10})
	vector_sequence_attention, sequence_attention_weights = unidir(Variable(torch.randn(2,5,6)), Variable(torch.randn(1,20).expand(2,20)),\
		Variable(utils.sequence_mask(torch.LongTensor([5,3]))), softmax=True)
	print(vector_sequence_attention)
	print(sequence_attention_weights)
