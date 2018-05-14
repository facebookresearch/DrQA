import torch
import torch.nn as nn
from torch.autograd import Variable

from . import utils1
from . import similarity
from . import unidirattention
from .unidirattention import UniDirAttention
from .unidirattention import UniDirAttentionItr

class SelfAttention(nn.Module):

	def __init__(self, dict_args):
		super(SelfAttention, self).__init__()
		self.attention_function = None
		if dict_args['similarity_function'] == 'ProjectionSimilaritySharedWeights':
			self.attention_function = UniDirAttention({'similarity_function': dict_args['similarity_function'], \
				'similarity_function_pointer': dict_args['similarity_function_pointer']})
		else:
			self.attention_function = UniDirAttention({'similarity_function': dict_args['similarity_function'], 'sequence1_dim': dict_args['sequence_dim'], \
				'sequence2_dim':dict_args['sequence_dim'], 'projection_dim':dict_args['projection_dim']})

	def forward(self, sequence, sequence_mask):
		#sequence: batch_size*num_words*iembed

		sequence = sequence.permute(1,0,2) #sequence: num_words*batch_size*iembed
		num_words, batch_size, input_dim = sequence.size()

		sequence_sequence_selfattn = Variable(sequence.data.new(num_words,batch_size,input_dim).zero_())
		#sequence_selfattn_weights = None
		if not self.training: sequence_selfattn_weights = Variable(sequence.data.new(num_words,batch_size,num_words).zero_())
		#sequence_sequence_selfattn: num_words*batch_size*iembed
		#sequence_selfattn_weights: num_words*batch_size*num_words

		print('add', sequence.size())
		for ith_item in range(num_words):
			vector = sequence[ith_item] #vector: batch_size*iembed
			vector_sequence_attention, sequence_attention_weights = self.attention_function(sequence.permute(1,0,2), vector, sequence_mask)
			#vector_sequence_attention: batch_size*iembed
			#sequence_attention_weights: batch_size*num_words
			sequence_sequence_selfattn[ith_item] = vector_sequence_attention
			if not self.training: sequence_selfattn_weights[ith_item] = sequence_attention_weights

		sequence_sequence_selfattn = sequence_sequence_selfattn.permute(1,0,2)
		if not self.training: sequence_selfattn_weights = sequence_selfattn_weights.permute(1,0,2)
		#sequence_mask: batch_size*num_words
		sequence_sequence_selfattn = utils1.mask_sequence(sequence_sequence_selfattn, sequence_mask)
		if not self.training: sequence_selfattn_weights = utils1.mask_sequence(sequence_selfattn_weights, sequence_mask)
		#sequence_sequence_selfattn: batch_size*num_words*iembed
		#sequence_selfattn_weights: batch_size*num_words*num_words
		if not self.training: return sequence_sequence_selfattn, sequence_selfattn_weights
		else: return sequence_sequence_selfattn, None
			
if __name__=='__main__':
	selfattn = SelfAttention({'similarity_function': 'WeightedSumProjection', 'sequence_dim':10, 'projection_dim':10})
	selfattn = selfattn.eval()
	sequence_sequence_selfattn, sequence_selfattn_weights =\
		selfattn(Variable(torch.randn(3,6,10)), Variable(utils1.sequence_mask(torch.LongTensor([5,3,6]))))
	print(sequence_sequence_selfattn)
	print(sequence_selfattn_weights)
