# -*- coding: utf-8 -*-
from collections import defaultdict
import random

import torch
from torch import nn
import torch.nn.functional as F

from ...utils.nn import get_rnn_hidden_state, get_activation_fn
from .. import FF
from ..attention import get_attention

import os
import sys
import numpy as np
from scipy.stats import binned_statistic

class ConditionalDecoder(nn.Module):
    """A conditional decoder with attention à la dl4mt-tutorial."""
    def __init__(self, input_size, hidden_size, ctx_size_dict, ctx_name, n_vocab,
                 rnn_type, tied_emb=False, dec_init='zero', dec_init_activ='tanh',
                 dec_init_size=None, att_type='mlp',
                 att_activ='tanh', att_bottleneck='ctx', att_temp=1.0,
                 att_ctx2hid=True,
                 transform_ctx=True, mlp_bias=False, dropout_out=0,
                 emb_maxnorm=None, emb_gradscale=False, sched_sample=0,
                 bos_type='emb', bos_dim=None, bos_activ=None, bos_bias=False,
                 out_logic='simple', emb_interact=None, emb_interact_dim=None,
                 emb_interact_activ=None, dec_inp_activ=None):
        super().__init__()

        # Normalize case
        self.rnn_type = rnn_type.upper()
        self.out_logic = out_logic
        # A persistent dictionary to save activations for further debugging
        # Currently only used in MMT decoder
        self.persistence = defaultdict(list)

        # Safety checks
        assert self.rnn_type in ('GRU', 'LSTM'), \
            "rnn_type '{}' not known".format(rnn_type)
        assert bos_type in ('emb', 'feats', 'zero'), "Unknown bos_type"
        assert dec_init.startswith(('zero', 'feats', 'mean_ctx', 'max_ctx', 'last_ctx')), \
            "dec_init '{}' not known".format(dec_init)

        RNN = getattr(nn, '{}Cell'.format(self.rnn_type))
        # LSTMs have also the cell state
        self.n_states = 1 if self.rnn_type == 'GRU' else 2

        # Set custom handlers for GRU/LSTM
        if self.rnn_type == 'GRU':
            self._rnn_unpack_states = lambda x: x
            self._rnn_pack_states = lambda x: x
        elif self.rnn_type == 'LSTM':
            self._rnn_unpack_states = self._lstm_unpack_states
            self._rnn_pack_states = self._lstm_pack_states

        # Set decoder initializer
        self._init_func = getattr(self, '_rnn_init_{}'.format(dec_init))

        # Other arguments
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size_dict = ctx_size_dict
        self.ctx_name = ctx_name
        self.n_vocab = n_vocab
        self.tied_emb = tied_emb
        self.dec_init = dec_init
        self.dec_init_size = dec_init_size
        self.dec_init_activ = dec_init_activ
        self.att_bottleneck = att_bottleneck
        self.att_activ = att_activ
        self.att_type = att_type
        self.att_temp = att_temp
        self.transform_ctx = transform_ctx
        self.att_ctx2hid = att_ctx2hid
        self.mlp_bias = mlp_bias
        self.dropout_out = dropout_out
        self.emb_maxnorm = emb_maxnorm
        self.emb_gradscale = emb_gradscale
        self.sched_sample = sched_sample
        self.bos_type = bos_type
        self.bos_dim = bos_dim
        self.bos_activ = bos_activ
        self.bos_bias = bos_bias
        self.emb_interact = emb_interact
        self.emb_interact_dim = emb_interact_dim
        self.emb_interact_activ = emb_interact_activ
        self.dec_inp_activ_fn = get_activation_fn(dec_inp_activ)

        # no-ops
        self.emb_fn = lambda e, f: e
        self.v_emb = None

        if self.emb_interact and self.emb_interact.startswith('trg'):
            self.ff_feats = FF(
                self.emb_interact_dim, self.input_size, bias=True,
                activ=self.emb_interact_activ)
            if self.emb_interact == 'trgmul':
                self.emb_fn = lambda e, f: e * f
            elif self.emb_interact == 'trgsum':
                self.emb_fn = lambda e, f: e + f
            self.emb_interact = True
        else:
            self.emb_interact = False

        if self.bos_type == 'feats':
            # Learn a visual <bos> embedding
            self.ff_bos = FF(self.bos_dim, self.input_size, bias=self.bos_bias,
                             activ=self.bos_activ)

        # Create target embeddings
        self.emb = nn.Embedding(self.n_vocab, self.input_size,
                                padding_idx=0, max_norm=self.emb_maxnorm,
                                scale_grad_by_freq=self.emb_gradscale)

        # Create attention layer
        Attention = get_attention(self.att_type)
        self.att = Attention(
            512,#self.ctx_size_dict[self.ctx_name],
            self.hidden_size,
            transform_ctx=self.transform_ctx,
            ctx2hid=self.att_ctx2hid,
            mlp_bias=self.mlp_bias,
            att_activ=self.att_activ,
            att_bottleneck=self.att_bottleneck,
            temp=self.att_temp)

        if self.dec_init != 'zero':
            # For source-based inits, input size is the encoding size
            # For 'feats', it's given by dec_init_size, no need to infer
            if self.dec_init.endswith('_ctx'):
                self.dec_init_size = self.ctx_size_dict[self.ctx_name]
            # Add a FF layer for decoder initialization
            self.ff_dec_init = FF(
                512,#self.dec_init_size,
                self.hidden_size * self.n_states,
                activ=self.dec_init_activ)

        # Create decoders
        self.dec0 = RNN(self.input_size, self.hidden_size)
        self.dec1 = RNN(self.hidden_size, self.hidden_size)

        # Output dropout
        if self.dropout_out > 0:
            self.do_out = nn.Dropout(p=self.dropout_out)

        # Output bottleneck: maps hidden states to target emb dim
        # simple: tanh(W*h)
        #   deep: tanh(W*h + U*emb + V*ctx)
        out_inp_size = self.hidden_size

        # Dummy op to return back the hidden state for simple output
        self.out_merge_fn = lambda h, e, c: h

        if self.out_logic == 'deep':
            out_inp_size += (self.input_size + self.hidden_size)
            self.out_merge_fn = lambda h, e, c: torch.cat((h, e, c), dim=1)

        # Final transformation that receives concatenated outputs or only h
        self.hid2out = FF(out_inp_size, self.input_size,
                          bias_zero=True, activ='tanh')

        # Final softmax
        self.out2prob = FF(self.input_size, self.n_vocab)

        # Tie input embedding matrix and output embedding matrix
        if self.tied_emb:
            self.out2prob.weight = self.emb.weight

        self.nll_loss = nn.NLLLoss(reduction="sum", ignore_index=0)

    def _lstm_pack_states(self, h):
        """Pack LSTM hidden and cell state."""
        return torch.cat(h, dim=-1)

    def _lstm_unpack_states(self, h):
        """Unpack LSTM hidden and cell state to tuple."""
        return torch.split(h, self.hidden_size, dim=-1)

    def _rnn_init_zero(self, ctx_dict):
        """Zero initialization."""
        ctx, _ = ctx_dict[self.ctx_name]
        return torch.zeros(
            ctx.shape[1], self.hidden_size * self.n_states, device=ctx.device)

    def _rnn_init_mean_ctx(self, ctx_dict):
        """Initialization with mean-pooled source annotations."""
        ctx, ctx_mask = ctx_dict[self.ctx_name]
        return self.ff_dec_init(
            ctx.sum(0).div(ctx_mask.unsqueeze(-1).sum(0))
            if ctx_mask is not None else ctx.mean(0))

    def _rnn_init_max_ctx(self, ctx_dict):
        """Initialization with max-pooled source annotations."""
        ctx, ctx_mask = ctx_dict[self.ctx_name]
        # Max-pooling may not care about mask (depends on non-linearity maybe)
        return self.ff_dec_init(ctx.max(0)[0])

    def _rnn_init_last_ctx(self, ctx_dict):
        """Initialization with the last source annotation."""
        ctx, ctx_mask = ctx_dict[self.ctx_name]
        if ctx_mask is None:
            h_0 = self.ff_dec_init(ctx[-1])
        else:
            last_idxs = ctx_mask.sum(0).sub(1).long()
            h_0 = self.ff_dec_init(ctx[last_idxs, range(ctx.shape[1])])
        return h_0

    def _rnn_init_feats(self, ctx_dict):
        """Feature based decoder initialization."""
        return self.ff_dec_init(ctx_dict['feats'][0].squeeze(0))

    def get_emb(self, idxs, tstep=-1):
        """Returns time-step based embeddings."""
        if tstep == 0:
            if self.bos_type == 'zero':
                # Constant-zero <bos> embedding
                return torch.zeros(
                    idxs.shape[0], self.input_size, device=idxs.device)
            elif self.bos_type == 'feats':
                # Feature-based <bos> computed in f_init()
                return self.bos
        # For other timesteps, look up the embedding layer
        if self.emb_interact and idxs.ndimension() == 1:
                # beam search
                embs = self.emb(idxs)
                return self.emb_fn(
                    embs.view((self.v_emb.shape[0], -1, self.v_emb.shape[1])),
                    self.v_emb.unsqueeze(1)).view(embs.shape)
        else:
            return self.emb_fn(self.emb(idxs), self.v_emb)

    def f_init(self, ctx_dict):
        """Returns the initial h_0 for the decoder."""
        self.history = defaultdict(list)
        # Compute <bos> out of 'feats' if requested
        if self.bos_type == 'feats':
            self.bos = self.ff_bos(ctx_dict['feats'][0]).squeeze(0)
        if self.emb_interact:
            # We have features for embedding interaction
            self.v_emb = self.ff_feats(ctx_dict['feats'][0]).squeeze(0)
        return self._init_func(ctx_dict)

    def f_next(self, ctx_dict, y, h):
        """Applies one timestep of recurrence."""
        # Get hidden states from the first decoder (purely cond. on LM)
        h1_c1 = self.dec0(y, self._rnn_unpack_states(h))
        h1 = get_rnn_hidden_state(h1_c1)

        # Apply attention
        txt_alpha_t, txt_z_t = self.att(
            h1.unsqueeze(0), *ctx_dict[self.ctx_name])

        if not self.training:
            self.history['alpha_txt'].append(txt_alpha_t)

        # Run second decoder (h1 is compatible now as it was returned by GRU)
        # Additional optional transformation is to make the comparison
        # fair with the MMT model.
        h2_c2 = self.dec1(self.dec_inp_activ_fn(txt_z_t), h1_c1)
        h2 = get_rnn_hidden_state(h2_c2)

        # Output logic
        logit = self.hid2out(self.out_merge_fn(h2, y, txt_z_t))

        # Apply dropout if any
        if self.dropout_out > 0:
            logit = self.do_out(logit)

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        log_p = F.log_softmax(self.out2prob(logit), dim=-1)

        # Return log probs and new hidden states
        return log_p, self._rnn_pack_states(h2_c2)

    def forward(self, ctx_dict, y):
        """Computes the softmax outputs given source annotations `ctx_dict[self.ctx_name]`
        and ground-truth target token indices `y`. Only called during training.

        Arguments:
            ctx_dict(dict): A dictionary of tensors that should at least contain
                the key `ctx_name` as the main source representation of shape
                S*B*ctx_dim`.
            y(Tensor): A tensor of `T*B` containing ground-truth target
                token indices for the given batch.
        """

        loss = 0.0

        # Get initial hidden state
        h = self.f_init(ctx_dict)

        # are we doing scheduled sampling?
        sched = self.training and (random.random() > (1 - self.sched_sample))

        # Convert token indices to embeddings -> T*B*E
        # Skip <bos> now
        bos = self.get_emb(y[0], 0)
        if self.visualize:
            log_p, h, h3_att_wts, h2_img_txt, h2_aud_txt, h1_txt, h1_img, h1_aud = self.f_next(ctx_dict, bos, h)
        else:
            log_p, h = self.f_next(ctx_dict, bos, h)
        loss += self.nll_loss(log_p, y[1])
        y_emb = self.get_emb(y[1:])

        if self.visualize:
            # Initialize the att wts array
            h3_att_arr = h3_att_wts.reshape(2,1)
            h2_img_txt = h2_img_txt.reshape(2,1)
            h2_aud_txt = h2_aud_txt.reshape(2,1)

            # Multiply the level 2 att with the wts from level 3 att
            h2_att_img_txt = h2_img_txt * h3_att_wts[0]
            h2_att_aud_txt = h2_aud_txt * h3_att_wts[1]
            #h2_att_arr = torch.tensor([(h2_img_txt[0][0]+h2_aud_txt[0][0])/2, h2_img_txt[0][1], h2_aud_txt[0][1]]).unsqueeze(0)

            # Multiply the level 1 att with the multiplied level 2 att
            h1_att_txt = h1_txt * ((h2_att_img_txt[0][0] + h2_att_aud_txt[0][0]) / 2)
            h1_att_img = h1_img * h2_att_img_txt[1][0]
            h1_att_aud = h1_aud * h2_att_aud_txt[1][0]

            # Binning the audio features which converts array to numpy
            bin_size = int(h1_att_aud.size(0)/30)
            h1_att_aud = h1_att_aud.squeeze(1).detach().cpu().numpy()
            h1_att_aud = binned_statistic(h1_att_aud, h1_att_aud, bins=bin_size)[0]
            h1_att_aud = h1_att_aud.reshape(-1,1)

        for t in range(y_emb.shape[0] - 1):
            emb = self.emb(log_p.argmax(1)) if sched else y_emb[t]
            if self.visualize:
                log_p, h, h3_att_wts, h2_img_txt, h2_aud_txt, h1_txt, h1_img, h1_aud = self.f_next(ctx_dict, emb, h)
                loss += self.nll_loss(log_p, y[t + 2])
                h3_att_arr = torch.cat((h3_att_arr, h3_att_wts.reshape(2,1)), dim=1)
                h2_img_txt = h2_img_txt.reshape(2,1) * h3_att_wts[0]
                h2_aud_txt = h2_aud_txt.reshape(2,1) * h3_att_wts[1]
                #att_t = torch.tensor([(h2_img_txt[0][0]+h2_aud_txt[0][0])/2, h2_img_txt[0][1], h2_aud_txt[0][1]]).unsqueeze(0)
                h2_att_img_txt = torch.cat((h2_att_img_txt, h2_img_txt), dim=1)
                h2_att_aud_txt = torch.cat((h2_att_aud_txt, h2_aud_txt), dim=1)
                h1_txt = h1_txt * ((h2_img_txt[0][0] + h2_aud_txt[0][0]) / 2)
                h1_img = h1_img * h2_img_txt[1][0]
                h1_aud = h1_aud * h2_aud_txt[1][0]
                h1_aud = binned_statistic(h1_aud.squeeze(1).detach().cpu().numpy(), h1_aud.squeeze(1).detach().cpu().numpy(), bins=bin_size)[0]
                h1_aud = h1_aud.reshape(-1,1)
                h1_att_txt = torch.cat((h1_att_txt, h1_txt), dim=1)
                h1_att_img = torch.cat((h1_att_img, h1_img), dim=1)
                h1_att_aud = np.concatenate((h1_att_aud, h1_aud), axis=1)
            else:
                log_p, h = self.f_next(ctx_dict, emb, h)
                loss += self.nll_loss(log_p, y[t + 2])

        if self.visualize:
            # Get the idx of the file being processed
            with open('/home/aman_khullar/att_wts/idx.txt', 'r') as f:
                idx = int(f.read().strip())

            with open('/home/aman_khullar/how2/text_300/sum_devtest_300/desc.tok.txt', 'r') as f:
                summaries = f.readlines()
                filename = summaries[idx].split(' ')[0]
                ref_txt = ' '.join(summaries[idx].split(' ')[1:])

            dirname = os.path.join('/home/aman_khullar/att_wts', filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            with open(os.path.join(dirname, 'h3_att.npy'), 'wb') as f:
                np.save(f, h3_att_arr.detach().cpu().numpy())
            with open(os.path.join(dirname, 'h2_att_img_txt.npy'), 'wb') as f:
                np.save(f, h2_att_img_txt.detach().cpu().numpy())
            with open(os.path.join(dirname, 'h2_att_aud_txt.npy'), 'wb') as f:
                np.save(f, h2_att_aud_txt.detach().cpu().numpy())
            with open(os.path.join(dirname, 'h1_att_txt.npy'), 'wb') as f:
                np.save(f, h1_att_txt.detach().cpu().numpy())
            with open(os.path.join(dirname, 'h1_att_img.npy'), 'wb') as f:
                np.save(f, h1_att_img.detach().cpu().numpy())
            with open(os.path.join(dirname, 'h1_att_aud.npy'), 'wb') as f:
                np.save(f, h1_att_aud)
            with open(os.path.join(dirname, 'ref.txt'), 'w') as f:
                f.write(ref_txt)

        return {'loss': loss}
