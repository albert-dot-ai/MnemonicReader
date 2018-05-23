#!/usr/bin/env python3
# Copyright 2018-present, HKUST-KnowComp.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Implementation of the Mnemonic Reader."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
from torch.autograd import Variable

from allennlp.modules.elmo import Elmo, _ElmoCharacterEncoder, batch_to_ids
from allennlp.nn.util import add_sentence_boundary_token_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# ------------------------------------------------------------------------------
# Network
# ------------------------------------------------------------------------------


class MnemonicReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}
    def __init__(self, args, normalize=True, tokens=None):
        super(MnemonicReader, self).__init__()
        # Store config
        self.args = args

        # Word embeddings (+1 for padding)
        self.embedding = nn.Embedding(args.vocab_size,
                                      args.embedding_dim,
                                      padding_idx=0)

        # Char embeddings (+1 for padding)
        self.char_embedding = nn.Embedding(args.char_size,
                                      args.char_embedding_dim,
                                      padding_idx=0)

        # Char rnn to generate char features
        self.char_rnn = layers.StackedBRNN(
            input_size=args.char_embedding_dim,
            hidden_size=args.char_hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        # TODO: move this to args
        self.init_elmo_embeds(tokens)

        elmo_layers = 2
        elmo_embedding_dim = 1024

        # doc_input_size = args.embedding_dim + args.char_hidden_size * 2 + elmo_embedding_dim * elmo_layers + args.num_features
        doc_input_size = elmo_embedding_dim * elmo_layers + args.num_features

        self.elmo_embedding = Elmo(options_file, weight_file, elmo_layers, dropout=0)

        # Encoder
        self.encoding_rnn = layers.StackedBRNN(
            input_size=doc_input_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            dropout_rate=args.dropout_rnn,
            dropout_output=args.dropout_rnn_output,
            concat_layers=False,
            rnn_type=self.RNN_TYPES[args.rnn_type],
            padding=args.rnn_padding,
        )

        doc_hidden_size = 2 * args.hidden_size
        
        # Interactive aligning, self aligning and aggregating
        self.interactive_aligners = nn.ModuleList()
        self.interactive_SFUs = nn.ModuleList()
        self.self_aligners = nn.ModuleList()
        self.self_SFUs = nn.ModuleList()
        self.aggregate_rnns = nn.ModuleList()
        for i in range(args.hop):
            # interactive aligner
            self.interactive_aligners.append(layers.SeqAttnMatch(doc_hidden_size, identity=True))
            self.interactive_SFUs.append(layers.SFU(doc_hidden_size, 3 * doc_hidden_size))
            # self aligner
            self.self_aligners.append(layers.SelfAttnMatch(doc_hidden_size, identity=True, diag=False))
            self.self_SFUs.append(layers.SFU(doc_hidden_size, 3 * doc_hidden_size))
            # aggregating
            self.aggregate_rnns.append(
                layers.StackedBRNN(
                    input_size=doc_hidden_size,
                    hidden_size=args.hidden_size,
                    num_layers=1,
                    dropout_rate=args.dropout_rnn,
                    dropout_output=args.dropout_rnn_output,
                    concat_layers=False,
                    rnn_type=self.RNN_TYPES[args.rnn_type],
                    padding=args.rnn_padding,
                )
            )

        # Memmory-based Answer Pointer
        self.mem_ans_ptr = layers.MemoryAnsPointer(
            x_size=2*args.hidden_size, 
            y_size=2*args.hidden_size, 
            hidden_size=args.hidden_size, 
            hop=args.hop,
            dropout_rate=args.dropout_rnn,
            normalize=normalize
        )

    def init_elmo_embeds(self, tokens):
        elmo_char_encoder = _ElmoCharacterEncoder(options_file, weight_file).cuda()

        # Init with 0s for NULL, UNK
        all_token_embeds = torch.zeros(2, 512).cuda()
        bos_token_embed = None
        eos_token_embed = None

        for i in range(0, len(tokens), 1000):
            token_char_ids = batch_to_ids([tokens[i:min(i+1000, len(tokens))]]).cuda()
            token_embeds = elmo_char_encoder(token_char_ids)['token_embedding']

            if bos_token_embed is None:
                bos_token_embed = token_embeds.data[0][0]
                eos_token_embed = token_embeds.data[0][-1]

            # Strip BOS, EOS embeddings
            all_token_embeds = torch.cat((all_token_embeds, token_embeds.data[0][1:-1]))

        all_token_embeds = torch.cat((all_token_embeds, bos_token_embed.unsqueeze(0), eos_token_embed.unsqueeze(0)))
        self.token_embedding_weights = torch.nn.Parameter(
            all_token_embeds, requires_grad = False,
        ).cuda()
        self.bos_id = all_token_embeds.size()[0] - 2
        self.eos_id = all_token_embeds.size()[0] - 1

    def forward(self, x1, x1_c, x1_f, x1_mask, x2, x2_c, x2_f, x2_mask, x1_texts, x2_texts):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_c = document char indices           [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_c = document char indices           [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x2_mask = question padding mask        [batch * len_q]
        x1_texts = document tokens
        x2 texts = question tokens
        """
        # Embed both document and question
        # x1_emb = self.embedding(x1)
        # x2_emb = self.embedding(x2)
        # x1_c_emb = self.char_embedding(x1_c)
        # x2_c_emb = self.char_embedding(x2_c)

        # Embed document and question text using elmo (batch_size, 3, num_timesteps, 1024)
        x1_with_bos_eos, x1_mask_with_bos_eos = add_sentence_boundary_token_ids(
            x1,
            x1_mask,
            self.bos_id,
            self.eos_id,
        )
        x2_with_bos_eos, x2_mask_with_bos_eos = add_sentence_boundary_token_ids(
            x2,
            x2_mask,
            self.bos_id,
            self.eos_id,
        )
        x1_token_embs = torch.nn.functional.embedding(
            x1_with_bos_eos,
            self.token_embedding_weights,
        )
        x1_token_embs = {
            'token_embedding': x1_token_embs,
            'mask': x1_mask_with_bos_eos,
        }
        x2_token_embs = torch.nn.functional.embedding(
            x2_with_bos_eos,
            self.token_embedding_weights,
        )
        x2_token_embs = {
            'token_embedding': x2_token_embs,
            'mask': x2_mask_with_bos_eos,
        }
        x1_elmo_embs = self.elmo_embedding(x1_token_embs)['elmo_representations']
        x2_elmo_embs = self.elmo_embedding(x2_token_embs)['elmo_representations']

        # Dropout on embeddings
        '''
        if self.args.dropout_emb > 0:
            x1_emb = F.dropout(x1_emb, p=self.args.dropout_emb, training=self.training)
            x2_emb = F.dropout(x2_emb, p=self.args.dropout_emb, training=self.training)
            x1_c_emb = F.dropout(x1_c_emb, p=self.args.dropout_emb, training=self.training)
            x2_c_emb = F.dropout(x2_c_emb, p=self.args.dropout_emb, training=self.training)
        '''

        # Generate char features
        '''
        x1_c_features = self.char_rnn(x1_c_emb, x1_mask)
        x2_c_features = self.char_rnn(x2_c_emb, x2_mask)
        '''

        # Combine input
        '''
        crnn_input = [x1_emb, *x1_elmo_embs, x1_c_features]
        qrnn_input = [x2_emb, *x2_elmo_embs, x2_c_features]
        '''
        crnn_input = x1_elmo_embs
        qrnn_input = x2_elmo_embs

        # Add manual features
        if self.args.num_features > 0:
            crnn_input.append(x1_f)
            qrnn_input.append(x2_f)

        # Encode document with RNN
        c = self.encoding_rnn(torch.cat(crnn_input, 2), x1_mask)

        # Encode question with RNN
        q = self.encoding_rnn(torch.cat(qrnn_input, 2), x2_mask)

        # Align and aggregate
        c_check = c
        for i in range(self.args.hop):
            q_tilde = self.interactive_aligners[i].forward(c_check, q, x2_mask)
            c_bar = self.interactive_SFUs[i].forward(c_check, torch.cat([q_tilde, c_check * q_tilde, c_check - q_tilde], 2))
            c_tilde = self.self_aligners[i].forward(c_bar, x1_mask)
            c_hat = self.self_SFUs[i].forward(c_bar, torch.cat([c_tilde, c_bar * c_tilde, c_bar - c_tilde], 2))
            c_check = self.aggregate_rnns[i].forward(c_hat, x1_mask)

        # Predict
        start_scores, end_scores = self.mem_ans_ptr.forward(c_check, q, x1_mask, x2_mask)
        
        return start_scores, end_scores
