import math
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable

from . import utils1
#import utils


class DotProductSimilarity(nn.Module):
	def __init__(self, dict_args):
		super(DotProductSimilarity, self).__init__()

	def forward(self, sequence1, sequence2):
		#sequence1: _*iembed
		#sequence2: _*iembed
		similarity_matrix = (sequence1*sequence2).sum(dim=-1) #similarity_matrix: _
		return similarity_matrix


class LinearConcatenationSimilarity(nn.Module):
	def __init__(self, dict_args):
		super(LinearConcatenationSimilarity, self).__init__()
		self.input_dim = dict_args['similarity_input_dim']

		self.weights = nn.Parameter(torch.Tensor(2*self.input_dim))
		self.bias = nn.Parameter(torch.Tensor(1))

		self.init_weights()

	def init_weights(self):
		stdv = 1.0 / math.sqrt(self.weights.size(-1))
		self.weights.data.uniform_(-stdv, stdv)
		self.bias.data.fill_(0)

	def forward(self, sequence1, sequence2):
		#sequence1: _*iembed
		#sequence2: _*iembed
		sequence = torch.cat((sequence1,sequence2),dim=-1) #sequence: _i*2.iembed
		similarity_matrix = torch.matmul(sequence,self.weights) + self.bias #similarity_matrix: _
		return similarity_matrix


class LinearConcatenationDotSimilarity(nn.Module):
	def __init__(self, dict_args):
		super(LinearConcatenationDotSimilarity, self).__init__()
		self.input_dim = dict_args['similarity_input_dim']

		self.weights = nn.Parameter(torch.Tensor(3*self.input_dim))
		self.bias = nn.Parameter(torch.Tensor(1))

		self.init_weights()

	def init_weights(self):
		stdv = 1.0 / math.sqrt(self.weights.size(-1))
		self.weights.data.uniform_(-stdv, stdv)
		self.bias.data.fill_(0)

	def forward(self, sequence1, sequence2):
		#sequence1: _*iembed
		#sequence2: _*iembed
		sequence = torch.cat((sequence1,sequence2,sequence1*sequence2),dim=-1) #sequence: _i*3.iembed
		similarity_matrix = torch.matmul(sequence,self.weights) + self.bias #similarity_matrix: _
		return similarity_matrix


class LinearProjectionSimilarity(nn.Module):
	def __init__(self, dict_args):
		super(LinearProjectionSimilarity, self).__init__()
		self.sequence1_dim = dict_args['sequence1_dim']
		self.sequence2_dim = dict_args['sequence2_dim']
		self.projection_dim = dict_args['projection_dim']

		self.sequence1_weights = nn.Parameter(torch.Tensor(self.projection_dim, self.sequence1_dim))
		self.sequence2_weights = nn.Parameter(torch.Tensor(self.projection_dim, self.sequence2_dim))
		self.sequence_bias = nn.Parameter(torch.Tensor(self.projection_dim,1))
		self.weights = nn.Parameter(torch.Tensor(self.projection_dim,1))
		self.bias = nn.Parameter(torch.Tensor(1))
		self.init_weights()

	def init_weights(self):
		stdv = 1.0 / math.sqrt(self.weights.size(-1))
		self.weights.data.uniform_(-stdv, stdv)
		self.bias.data.fill_(0)
		self.sequence_bias.data.fill_(0)
		stdv = 1.0 / math.sqrt(self.sequence1_weights.size(1))
		self.sequence1_weights.data.uniform_(-stdv, stdv)
		stdv = 1.0 / math.sqrt(self.sequence2_weights.size(1))
		self.sequence2_weights.data.uniform_(-stdv, stdv)

	def forward(self, sequence1, sequence2):
		#sequence1: _*sequence1_dim
		#sequence2: _*sequence2_dim
		if sequence1.dim() == 3 and sequence2.dim() == 3:
			sequence1 = sequence1.permute(0,2,1) #sequence1(context): batch_size*sequence1_dim*num_words1
			sequence2 = sequence2.permute(0,2,1) #sequence2(query): batch_size*sequence2_dim*num_words1
		elif sequence1.dim() == 4 and sequence2.dim() == 4:
			sequence1 = sequence1.permute(0,1,3,2) #sequence1: batch_size*num_words1*sequence1_dim*num_words2
			sequence2 = sequence2.permute(0,1,3,2) #sequence2: batch_size*num_words1*sequence2_dim*num_words2
		
		sequence1_projection = torch.matmul(self.sequence1_weights,sequence1) #sequence1_projection: _*projection_dim*num_words
		sequence2_projection = torch.matmul(self.sequence2_weights,sequence2) #sequence1_projection: _*projection_dim*num_words
		similarity_projection = sequence1_projection + sequence2_projection + self.sequence_bias #similarity_projection: _*projection_dim*num_words
		similarity_projection = functional.tanh(similarity_projection) #similarity_projection: _*projection_dim*num_words
		similarity_matrix = torch.matmul((self.weights).t(), similarity_projection) + self.bias #similarity_matrix: _*1*num_words
		similarity_matrix = similarity_matrix.squeeze(-2) #similarity_matrix: batch_size*num_words1 or batch_size*num_words1*num_words2
		return similarity_matrix

class ProjectionSimilaritySharedWeights(nn.Module):
	def __init__(self, dict_args):
		super(ProjectionSimilaritySharedWeights, self).__init__()
		self.projection_dim = dict_args['projection_dim']

		self.sequence1_weights = dict_args['sequence1_weights']
		self.sequence2_weights = dict_args['sequence2_weights']
		if 'sequence3_weights' in dict_args:
			self.sequence3_weights = dict_args['sequence3_weights']		
		self.weights = dict_args['weights']
		
		self.sequence_bias = nn.Parameter(torch.Tensor(self.projection_dim,1))
		self.bias = nn.Parameter(torch.Tensor(1))
		self.init_weights()

	def init_weights(self):
		self.bias.data.fill_(0)
		self.sequence_bias.data.fill_(0)

	def forward(self, sequence1, sequence2, sequence3=None):
		#sequence1: _*sequence1_dim
		#sequence2: _*sequence2_dim
		if sequence1.dim() == 3 and sequence2.dim() == 3:
			sequence1 = sequence1.permute(0,2,1) #sequence1(context): batch_size*sequence1_dim*num_words1
			sequence2 = sequence2.permute(0,2,1) #sequence2(query): batch_size*sequence2_dim*num_words1
			if sequence3 is not None:
				sequence3 = sequence3.permute(0,2,1) #sequence3(query2): batch_size*sequence3_dim*num_words1			
		elif sequence1.dim() == 4 and sequence2.dim() == 4:
			sequence1 = sequence1.permute(0,1,3,2) #sequence1: batch_size*num_words1*sequence1_dim*num_words2
			sequence2 = sequence2.permute(0,1,3,2) #sequence2: batch_size*num_words1*sequence2_dim*num_words2
			if sequence3 is not None:
				sequence3 = sequence3.permute(0,1,3,2) #sequence3: batch_size*num_words1*sequence3_dim*num_words2

		sequence1_projection = torch.matmul(self.sequence1_weights,sequence1) #sequence1_projection: _*projection_dim*num_words
		sequence2_projection = torch.matmul(self.sequence2_weights,sequence2) #sequence1_projection: _*projection_dim*num_words
		similarity_projection = sequence1_projection + sequence2_projection + self.sequence_bias #similarity_projection: _*projection_dim*num_words
		if sequence3 is not None:
			sequence3_projection = torch.matmul(self.sequence3_weights,sequence3) #sequence3_projection: _*projection_dim*num_words
			similarity_projection = similarity_projection + sequence3_projection
		similarity_projection = functional.tanh(similarity_projection) #similarity_projection: _*projection_dim*num_words
		similarity_matrix = torch.matmul((self.weights).t(), similarity_projection) + self.bias #similarity_matrix: _*1*num_words
		similarity_matrix = similarity_matrix.squeeze(-2) #similarity_matrix: batch_size*num_words1 or batch_size*num_words1*num_words2
		return similarity_matrix
