import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiDAFAttention(nn.Module):
    """
    Bidirectional attention computes attention in two directions:
    The text attends to the modality (image/audio) and the modality attends to the text.

    The output of this layer is the concatenation of:
    [text, text2image_attention, text * text2image_attention, text * image2text_attention] or
    [text, text2audio_attention, text * text2audio_attention, text * audio2text_attention]
    based on the modality used.

    This concatenation allows the attention vector at each timestep, along with the embeddings 
    from previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, text_length, 8 * hidden_size)

    Args:
        hidden_size (int) : Size of hidden activations.
        drop_prob (float) : Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.text_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.modality_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.text_modality_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.text_weight, self.modality_weight, self.text_modality_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, text, modality, text_mask, modality_mask):
        batch_size, text_length, _ = text.size()
        modality_length = modality.size(1)
        #print("Shape of text : {} \n Shape of modality is : {}".format(text.shape, modality.shape))
        s = self.get_similarity_matrix(text, modality)                     # (batch_size, text_length, modality_length)
        #print("Similarity matrix size : {}".format(s.size()))
        text_mask = text_mask.view(batch_size, text_length, 1)          # (batch_size, text_length, 1)
        modality_mask = modality_mask.view(batch_size, 1, modality_length)    # (batch_size, 1, modality_length)
        s1 = masked_softmax(s, modality_mask, dim=2)                       # (batch_size, text_length, modality_length)
        s2 = masked_softmax(s, text_mask, dim=1)                        # (batch_size, text_length, modality_length)

        # (batch_size, text_length, modality_length) x (batch_size, modality_length, hidden_size) => (batch_size, text_length, hidden_size)
        a = torch.bmm(s1, modality)

        # (batch_size, text_length, text_length) x (batch_size, text_length, hidden_size) => (batch_size, text_length, hidden_size) 
        b = torch.bmm(torch.bmm(s1, s2.transpose(1,2)), text)

        x = torch.cat([text, a, text * a, text * b], dim = 2)            # (batch_size, text_length, 4 * hidden_size)

        return x

    def get_similarity_matrix(self, text, modality):
        """
        Get the "similarity matrix" between text and the modality (image/audio).

        Concatenate the three vectors then project the result with a single weight matrix. This method is more
        memory-efficient implementation of the same operation.

        This is the Equation 1 of the BiDAF paper.
        """
        text_length, modality_length = text.size(1), modality.size(1)
        text = F.dropout(text, self.drop_prob, self.training)           # (batch_size, text_length, hidden_size)
        modality = F.dropout(modality, self.drop_prob, self.training)         # (batch_size, modality_length, hidden_size)

        # Shapes : (batch_size, text_length, modality_length)
        #print("text size is : {} \n text_weight size is : {} \n modality size is : {} \n modality_weight size is : {}, text_mod_weight size is  :{}".format(text.size(), self.text_weight.size(), modality.size(), self.modality_weight.size(), self.text_modality_weight.size()))
        s0 = torch.matmul(text, self.text_weight).expand([-1, -1, modality_length])
        s1 = torch.matmul(modality, self.modality_weight).transpose(1,2).expand([-1, text_length, -1])
        s2 = torch.matmul(text * self.text_modality_weight, modality.transpose(1,2))
        s = s0 + s1 + s2 + self.bias

        return s

def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.

    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.

    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs
    
