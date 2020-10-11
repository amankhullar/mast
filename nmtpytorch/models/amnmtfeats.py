# -*- coding: utf-8 -*-
import logging

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..datasets import MultimodalDataset
from ..layers import ConditionalMMDecoder, ConditionalMM3DecoderCoverage, TextEncoder
from .nmt import NMT
from ..layers import BiDAFAttention
from ..utils.data import sort_batch

logger = logging.getLogger('nmtpytorch')

class Embedding(torch.nn.Module):
    """
    Prejection of the audio and image features into hidden size

    Args:
        embedding_size (torch.Tensor) : Pre-trained modality embedding size.
        hidden_size (int) : Size of hidden activations.
        drop_prob (float) : Probability of zero-in out activations.
    """
    def __init__(self, embedding_size, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.proj = torch.nn.Linear(embedding_size, hidden_size, bias = False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = torch.nn.functional.dropout(x, self.drop_prob, self.training)  # (batch_size, seq_len, embed_size)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb

class HighwayEncoder(torch.nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = torch.nn.functional.relu(transform(x))
            x = g * t + (1 - g) * x

        return x

class RNNEncoder(torch.nn.Module):
    """
    General-purpose layer for encoding a sequence using a bidirectional RNN.

    This encoding is for the text input data. 
    The encoded output is the RNN's hidden state at each position,
    which has shape (batch_size, seq_len, hidden_size * 2).

    Args:
        input_size (int) : Size of a single timestep in the input (The number of expected features in the input element).
        hidden_size (int) : Size of the RNN hidden state.
        num_layers (int) : Number of layers of RNN cells to use.
        drop_prob (float) : Probability of zero-ing out activations.
    """
    def __init__(self, input_size, hidden_size, num_layers, drop_prob = 0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = torch.nn.LSTM(input_size, hidden_size, num_layers,
                           bidirectional = True,
                           dropout = drop_prob if num_layers > 1 else 0.)

    def forward(self, x):
        # Convert python list of lengths to a Tensor
        mask = None
        x_sb = torch.sum(x, -1)
        #print('x_sb size:', x_sb.size())
        oidxs, sort_idx, lengths, mask = sort_batch(x_sb)
        
        #mask = torch.ones_like(x, dtype=float)

        lengths = torch.LongTensor(lengths)
        #print("lengths size and type", lengths.size(), lengths.type())

        #print('oidxs:', oidxs)
        #print('sort_idx:', sort_idx)
        #print('lengths:', lengths)
        # Save the original padded length for use by pad_packed_sequence
        #orig_len = x.size(1)

        #print('x before sort:', x.size())
        x = x[:, sort_idx]    # (batch_size, seq_len, input_size)
        #print('x after sort:', x.size())
        x = pack_padded_sequence(x, lengths)
        #print('x after pack_padded_sequence:', x.data.size())

        # Apply RNN
        x, (x_hidden, _) = self.rnn(x) # (batch_size, seq_len, 2 * hidden_size)
        #print('x after rnn:', x.data.size())

        # Unpack and reverse sort
        #x, _ = pad_packed_sequence(x, total_length = orig_len)
        #_, unsort_idx = sort_idx.sort(0)
        #x = x[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)

        x = pad_packed_sequence(x)[0][:, oidxs]
        #print('x after pad_packed:', x.size())

        # Apply dropout (RNN applies after all but the last layer)
        x = torch.nn.functional.dropout(x, self.drop_prob, self.training)

        # x_hidden = x_hidden.transpose(0,1)          # to convert the hidden state to batch first

        return x, mask

class AttentiveMNMTFeatures(NMT):
    """An end-to-end sequence-to-sequence NMT model with visual attention over
    pre-extracted convolutional features.
    """
    def set_defaults(self):
        # Set parent defaults
        super().set_defaults()
        self.defaults.update({
            'fusion_type': 'concat',    # Multimodal context fusion (sum|mul|concat)
            'fusion_activ': None,       # Multimodal context non-linearity
            'n_channels': 2048,         # depends on the features used
            'alpha_c': 0.0,             # doubly stoch. attention
            'mm_att_type': 'md-dd',     # multimodal attention type
                                        # md: modality dep.
                                        # mi: modality indep.
                                        # dd: decoder state dep.
                                        # di: decoder state indep.
            'out_logic': 'simple',      # simple vs deep output
            'persistent_dump': False,   # To save activations during beam-search
            'img_sequence': False,      # if true img is sequence of img features,
                                        # otherwise it's a conv map
        })

    def __init__(self, opts):
        super().__init__(opts)
        if self.opts.model['alpha_c'] > 0:
            self.aux_loss['alpha_reg'] = 0.0

        # TODO: get the correct dimensions
        #self.aud_emb = Embedding(embedding_size=43,     # TODO : change img to aud : Since image embedding size is not 300, we need another highway encoder layer
        #                    hidden_size=256,                 # and we cannot increase the hidden size beyond 100
        #                     drop_prob=self.opts.model['dropout_ctx'])

        #self.img_emb = Embedding(embedding_size=2048,     # Since image embedding size is not 300, we need another highway encoder layer
        #                     hidden_size=256,                 # and we cannot increase the hidden size beyond 100
        #                     drop_prob=self.opts.model['dropout_ctx'])

        self.aud_enc = RNNEncoder(input_size=43, 
                                     hidden_size=256, 
                                     num_layers=1, 
                                     drop_prob=self.opts.model['dropout_ctx'])

        self.img_enc = RNNEncoder(input_size=2048,
                                    hidden_size=256,
                                    num_layers=1,
                                    drop_prob=self.opts.model['dropout_ctx'])

        #self.aud_enc = TextEncoder(
        #    input_size=43,
        #    hidden_size=self.opts.model['enc_dim'],
        #    n_vocab=self.n_src_vocab,
        #    rnn_type=self.opts.model['enc_type'],
        #    dropout_emb=self.opts.model['dropout_emb'],
        #    dropout_ctx=self.opts.model['dropout_ctx'],
        #    dropout_rnn=self.opts.model['dropout_enc'],
        #    num_layers=self.opts.model['n_encoders'],
        #    emb_maxnorm=self.opts.model['emb_maxnorm'],
        #    emb_gradscale=self.opts.model['emb_gradscale'])

        #self.img_enc = TextEncoder(
        #    input_size=2048,
        #    hidden_size=self.opts.model['enc_dim'],
        #    n_vocab=self.n_src_vocab,
        #    rnn_type=self.opts.model['enc_type'],
        #    dropout_emb=self.opts.model['dropout_emb'],
        #    dropout_ctx=self.opts.model['dropout_ctx'],
        #    dropout_rnn=self.opts.model['dropout_enc'],
        #    num_layers=self.opts.model['n_encoders'],
        #    emb_maxnorm=self.opts.model['emb_maxnorm'],
        #    emb_gradscale=self.opts.model['emb_gradscale'])

        # self.bidaf_att_audio = BiDAFAttention(hidden_size=2*self.opts.model['enc_dim'], 
        #                                       drop_prob=self.opts.model['dropout_ctx'])

        # self.bidaf_att_image = BiDAFAttention(hidden_size=2*self.opts.model['enc_dim'], 
        #                                       drop_prob=self.opts.model['dropout_ctx'])

        # self.mod_ta = RNNEncoder(input_size=8*self.opts.model['enc_dim'],
        #                         hidden_size=self.opts.model['enc_dim'],
        #                         num_layers=2,
        #                         drop_prob=self.opts.model['dropout_ctx'])

        # self.mod_ti = RNNEncoder(input_size=8*self.opts.model['enc_dim'],
        #                         hidden_size=self.opts.model['enc_dim'],
        #                         num_layers=2,
        #                         drop_prob=self.opts.model['dropout_ctx'])

    def setup(self, is_train=True):
        self.ctx_sizes['image'] = self.opts.model['n_channels']

        ########################
        # Create Textual Encoder
        ########################
        self.enc = TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
            rnn_type=self.opts.model['enc_type'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'])

        # self.enc_img_text = TextEncoder(
        #     input_size=8*self.opts.model['enc_dim'],
        #     hidden_size=self.opts.model['enc_dim'],
        #     n_vocab=self.n_src_vocab,
        #     rnn_type=self.opts.model['enc_type'],
        #     dropout_emb=self.opts.model['dropout_emb'],
        #     dropout_ctx=self.opts.model['dropout_ctx'],
        #     dropout_rnn=self.opts.model['dropout_enc'],
        #     num_layers=2,
        #     emb_maxnorm=self.opts.model['emb_maxnorm'],
        #     emb_gradscale=self.opts.model['emb_gradscale'])

        # self.enc_aud_text = TextEncoder(
        #     input_size=8*self.opts.model['enc_dim'],
        #     hidden_size=self.opts.model['enc_dim'],
        #     n_vocab=self.n_src_vocab,
        #     rnn_type=self.opts.model['enc_type'],
        #     dropout_emb=self.opts.model['dropout_emb'],
        #     dropout_ctx=self.opts.model['dropout_ctx'],
        #     dropout_rnn=self.opts.model['dropout_enc'],
        #     num_layers=2,
        #     emb_maxnorm=self.opts.model['emb_maxnorm'],
        #     emb_gradscale=self.opts.model['emb_gradscale'])

        # Create Decoder
        self.dec = ConditionalMM3DecoderCoverage(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.sl),
            fusion_type=self.opts.model['fusion_type'],
            fusion_activ=self.opts.model['fusion_activ'],
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            att_type=self.opts.model['att_type'],
            mm_att_type=self.opts.model['mm_att_type'],
            out_logic=self.opts.model['out_logic'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            att_ctx2hid=self.opts.model['att_ctx2hid'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            persistent_dump=self.opts.model['persistent_dump'])

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data[split + '_set'],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model.get('max_len', None),
            order_file=self.opts.data[split + '_set'].get('ord', None))
        logger.info(dataset)
        return dataset

    def encode(self, batch, **kwargs):
        # Let's start with a None mask by assuming that
        # we have a fixed-length feature collection
        txt_feats, txt_mask = self.enc(batch[self.sl])
        img_feats, img_mask = batch['image'], None
        aud_feats, aud_mask = batch['audio'], None          
        #print("text feats size : ", batch[self.sl].size())
        #print("image feats size: ", batch['image'].size())
        #print("audio feats size: ", batch['audio'].size())
        #emb_img_feats = self.img_emb(img_feats.transpose(0,1))
        #emb_aud_feats = self.aud_emb(aud_feats.transpose(0,1))
        #print("Emb images size : ", emb_img_feats.size())
        #print("Embedding done")
        #enc_img_feats, img_mask = self.img_enc(emb_img_feats)
        #enc_aud_feats, aud_mask = self.aud_enc(emb_aud_feats)

        enc_img_feats, img_mask = self.img_enc(img_feats)
        enc_aud_feats, aud_mask = self.aud_enc(aud_feats)

        #print('enc_img_feats, img_mask:', enc_img_feats.size(), img_mask.size())
        #print('enc_aud_feats, aud_mask:', enc_aud_feats.size(), aud_mask.size())

        if self.opts.model['img_sequence']:
            # Let's create mask in this case
            img_mask = img_feats.ne(0).float().sum(2).ne(0).float()
            aud_mask = aud_feats.ne(0).float().sum(2).ne(0).float()
        #print("Masks are - text : {} \t img : {}".format(txt_mask, img_mask))

        # Applying BIDAF
        # txt_aud_att = self.bidaf_att_audio(txt_feats.transpose(0,1), enc_aud_feats.transpose(0,1), txt_mask, aud_mask).transpose(0,1)   # (batch_size, num_sentences, 8 * hidden_size)
        # txt_img_att = self.bidaf_att_image(txt_feats.transpose(0,1), enc_img_feats.transpose(0,1), txt_mask, img_mask).transpose(0,1)   # (batch_size, num_sentences, 8 * hidden_size)
        #print("Reached here balle balle")

        #Modality encoding
        #mod_txt_aud, _ = self.enc_aud_text(txt_aud_att)                        # (batch_size, num_sentences, 2 * hidden_size)
        #mod_txt_img, _ = self.enc_img_text(txt_img_att)                        # (batch_size, num_sentences, 2 * hidden_size)
        # mod_txt_aud, _ = self.mod_ta(txt_aud_att)
        # mod_txt_img, _ = self.mod_ti(txt_img_att)

        # return {
        #     'image': (img_feats, img_mask),
        #     str(self.sl): self.enc(batch[self.sl]),
        # }

        #print("img feats size : {} \t txt_img_att size : {}".format(img_feats.size(), txt_img_att.size()))
        #print("self enc size : {}".format(self.enc(batch[self.sl])[0].size()))
        #print("img feats size : {} \t img_mask size : {} \t img_att size : {}".format(img_feats.size(), img_mask.size(), txt_img_att.size()))
        #print("aud feats size : {} \t aud_mask size : {} \t aud_att size : {}".format(aud_feats.size(), aud_mask.size(), txt_aud_att.size()))
        return {
            #'image_text': txt_img_att,
            #'audio_text': txt_aud_att,
            'image': (enc_img_feats, img_mask),
            'audio': (enc_aud_feats, aud_mask),
            # str(self.sl): self.enc(batch[self.sl]),
            str(self.sl): (txt_feats, txt_mask)
        }

    def forward(self, batch, **kwargs):
        result = super().forward(batch)

        if self.training and self.opts.model['alpha_c'] > 0:
            alpha_loss = (
                1 - torch.cat(self.dec.history['alpha_img']).sum(0)).pow(2).sum(0)
            self.aux_loss['alpha_reg'] = alpha_loss.mean().mul(
                self.opts.model['alpha_c'])

        return result
