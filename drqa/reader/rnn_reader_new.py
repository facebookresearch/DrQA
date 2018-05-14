#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the RNN based DrQA reader."""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from . import layers
from . import tcn
from . import self_attention
from .self_attention import SelfAttention
import time

# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout_rate, dropout_output):
        super(CustomLSTM, self).__init__()
        #self.num_layers = num_layers
        #self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.rnns = nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.dropout_output = dropout_output
        self.num_layers = num_layers
        for i in range(num_layers):
            if i==0:
                pass
            elif bidirectional:
                input_size = 2 * hidden_size
            else:
                input_size = hidden_size
            self.rnns.append(nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional))
            

    def forward(self, x):
        x = x.transpose(0,1)
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input, p=self.dropout_rate, training=self.training)
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)
        output = outputs[-1]
        output = output.transpose(0, 1)
        return output.contiguous()



class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_levels, dropout=0.3,
                 kernel_size=2, bidirectional=False, concat_rnn_layers=False, norm="weight", affine=False):
        super(TCN, self).__init__()
        num_channels = [output_size] * num_levels
        self.tcn = tcn.TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout, concat_layers=concat_rnn_layers, norm=norm, affine=affine)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.reverse = tcn.TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout, concat_layers=concat_rnn_layers, norm=norm, affine=affine)
            self.indices = Variable(torch.LongTensor([0])).cuda() #dummy variable for initializing
        

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        output = y.transpose(1, 2)
        if self.bidirectional:
            del self.indices
            self.indices = Variable(torch.LongTensor(list(reversed(range(x.shape[2]))))).cuda()
            x_reverse_torch = torch.index_select(x, 2, self.indices)
            y2 = self.reverse(x_reverse_torch)
            y2_reverse_torch = torch.index_select(y2, 2, self.indices)
            output = torch.cat((output, y2_reverse_torch.transpose(1,2)), 2)
        return output.contiguous()
            

class RnnDocReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__()
        # Store config
        self.attention = args.self_attention

        self.tcn_output_dim = 768
        if self.attention:
            self_matching_layer_args = {
                                            'similarity_function': 'WeightedSumProjection',
                                            'sequence_dim': self.tcn_output_dim, 
                                            'projection_dim': int(self.tcn_output_dim/2)
                                        }
            self.self_matching_layer = SelfAttention(self_matching_layer_args)


        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Projection for attention weighted question
        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)

        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb:
            doc_input_size += args.embedding_dim

        if args.rnn_type == 'custom_lstm':
            self.doc_rnn = CustomLSTM(doc_input_size, args.hidden_size, args.doc_layers, 
                                      args.bidirectional, args.dropout_rnn, args.dropout_rnn_output)
            self.question_rnn = CustomLSTM(args.embedding_dim, args.hidden_size, args.question_layers, 
                                           args.bidirectional, args.dropout_rnn, args.dropout_rnn_output)
        elif args.rnn_type == 'tcn':
            self.doc_rnn = TCN(doc_input_size, args.hidden_size, args.doc_layers, args.dropout_rnn, args.tcn_filter_size, args.bidirectional, args.concat_rnn_layers, args.norm, args.affine)
            self.question_rnn = TCN(args.embedding_dim, args.hidden_size, args.question_layers, args.dropout_rnn, args.tcn_filter_size, args.bidirectional, args.concat_rnn_layers, args.norm, args.affine)
        else:
            # RNN document encoder
            self.doc_rnn = layers.StackedBRNN(
                input_size=doc_input_size,
                hidden_size=args.hidden_size,
                num_layers=args.doc_layers,
                dropout_rate=args.dropout_rnn,
                dropout_output=args.dropout_rnn_output,
                concat_layers=args.concat_rnn_layers,
                rnn_type=self.RNN_TYPES[args.rnn_type],
                padding=args.rnn_padding,
            )
            # RNN question encoder
            self.question_rnn = layers.StackedBRNN(
                input_size=args.embedding_dim,
                hidden_size=args.hidden_size,
                num_layers=args.question_layers,
                dropout_rate=args.dropout_rnn,
                dropout_output=args.dropout_rnn_output,
                concat_layers=args.concat_rnn_layers,
                rnn_type=self.RNN_TYPES[args.rnn_type],
                padding=args.rnn_padding,
            )

        # Output sizes of rnn encoders
        if args.bidirectional:
            doc_hidden_size = 2 * args.hidden_size
            question_hidden_size = 2 * args.hidden_size
        else:
            doc_hidden_size = args.hidden_size
            question_hidden_size = args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers

        # Question merging
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)

        # Bilinear attention for span start/end
        self.start_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )
        self.end_attn = layers.BilinearSeqAttn(
            doc_hidden_size,
            question_hidden_size,
            normalize=normalize,
        )

    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        # Embed both document and question
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        # Dropout on embeddings
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb,
                                           training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb,
                                           training=self.training)

        # Form document encoding inputs
        drnn_input = [x1_emb]

        # Add attention-weighted question representation
        if self.args.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input.append(x2_weighted_emb)

        # Add manual features
        if self.args.num_features > 0:
            drnn_input.append(x1_f)
        
        #start_time = time.time()

        if self.args.rnn_type == 'custom_lstm':
            doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2))
            question_hiddens = self.question_rnn(x2_emb)
        elif self.args.rnn_type == 'tcn':
            doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2))
            question_hiddens = self.question_rnn(x2_emb)
        else:
            # Encode document with RNN
            doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask)

            # Encode question with RNN + merge hiddens
            question_hiddens = self.question_rnn(x2_emb, x2_mask)
        #end_time = time.time()
        #print(doc_hiddens.size())
        #print(end_time-start_time)
        if self.attention:
            doc_hiddens, self_match_weights = self.self_matching_layer(doc_hiddens, x1_mask)

        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

        # Predict start and end positions
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return start_scores, end_scores

