import math

import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_mlp(input_dim, hidden_dims, output_dim=None,
              use_batchnorm=False, dropout=0, fc_bias=True, relu=True):
    layers = []
    D = input_dim
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(input_dim))
    if hidden_dims:
        for dim in hidden_dims:
            layers.append(nn.Linear(D, dim, bias=fc_bias))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            if relu:
                layers.append(nn.ReLU(inplace=True))
            D = dim
    if output_dim:
        layers.append(nn.Linear(D, output_dim, bias=fc_bias))
    return nn.Sequential(*layers)


class SoftAttention(nn.Module):
    """Soft-Attention without learnable parameters
    """

    def __init__(self):
        super(SoftAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h, proj_context, context=None, mask=None, reverse_attn=False):
        """Propagate h through the network.

        h: batch x dim (concat(img, action))
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        """
        # Get attention
        attn = torch.bmm(proj_context, h.unsqueeze(2)).squeeze(2)  # batch x seq_len

        if reverse_attn:
            attn = -attn

        if mask is not None:
            attn.data.masked_fill_((mask == 0).data, -float('inf'))
        attn = self.softmax(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        if context is not None:
            weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        else:
            weighted_context = torch.bmm(attn3, proj_context).squeeze(1)  # batch x dim

        return weighted_context, attn


class PositionalEncoding(nn.Module):
    """Implement the PE function to introduce the concept of relative position"""

    def __init__(self, d_model, dropout, max_len=80):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 2i + 1
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def create_mask(batchsize, max_length, length):
    """Given the length create a mask given a padded tensor"""
    tensor_mask = torch.zeros(batchsize, max_length)
    for idx, row in enumerate(tensor_mask):
        row[:length[idx]] = 1
    return tensor_mask.to(device)

def proj_masking(feat, projector, mask=None):
    """Universal projector and masking"""
    proj_feat = projector(feat.view(-1, feat.size(2)))
    proj_feat = proj_feat.view(feat.size(0), feat.size(1), -1)
    if mask is not None:
        return proj_feat * mask.unsqueeze(2).expand_as(proj_feat)
    else:
        return proj_feat