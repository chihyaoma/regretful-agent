import numpy as np

import torch
import torch.nn as nn

from models.modules import build_mlp, SoftAttention, PositionalEncoding, create_mask, proj_masking


class Regretful(nn.Module):
    """
    The model for the regretful agent (CVPR 2019)

    The Regretful Agent: Heuristic-Aided Navigation through Progress Estimation
    GitHub: https://github.com/chihyaoma/regretful-agent
    Project: https://chihyaoma.github.io/project/2019/02/25/regretful.html
    """

    def __init__(self, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, rnn_dropout, max_len, fc_bias=True, max_navigable=16):
        super(Regretful, self).__init__()

        self.opts = opts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_navigable = max_navigable

        proj_navigable_kwargs = {
            'input_dim': img_feat_input_dim,
            'hidden_dims': img_fc_dim,
            'use_batchnorm': img_fc_use_batchnorm,
            'dropout': img_dropout,
            'fc_bias': fc_bias
        }
        self.proj_navigable_mlp = build_mlp(**proj_navigable_kwargs)

        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1])

        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=fc_bias)
        self.positional_encoding = PositionalEncoding(rnn_hidden_size, dropout=0.1, max_len=max_len)
        self.soft_attn = SoftAttention()

        self.dropout = nn.Dropout(p=rnn_dropout)
        self.lstm = nn.LSTMCell(img_fc_dim[-1] * 2 + rnn_hidden_size, rnn_hidden_size)

        self.logit_fc = nn.Linear(rnn_hidden_size * 2, img_fc_dim[-1])

        self.h2_fc_lstm = nn.Linear(rnn_hidden_size + img_fc_dim[-1], rnn_hidden_size, bias=fc_bias)

        self.critic_fc = nn.Linear(max_len + rnn_hidden_size, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.critic_valueDiff_fc = nn.Linear(1, 2)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

        self.move_fc = nn.Linear(img_fc_dim[-1], img_fc_dim[-1] + opts.tiled_len)

        self.num_predefined_action = 1

    def block_oscillation(self, batch_size, navigable_mask, oscillation_index, block_oscillation_index):

        navigable_mask[torch.LongTensor(range(batch_size)), np.array(oscillation_index) + 1] = \
        navigable_mask[torch.LongTensor(range(batch_size)), np.array(oscillation_index) + 1] * (1 - torch.Tensor(block_oscillation_index).to(self.device))

        assert (navigable_mask.sum(1) > 0).all(), "All actions are blocked ...? "

        return navigable_mask

    def forward(self, img_feat, navigable_feat, pre_feat, pre_value, h_0, c_0, ctx, navigable_index=None, navigable_idx_to_previous=None,
                oscillation_index=None, block_oscillation_index=None, ctx_mask=None, seq_lengths=None,
                prevent_oscillation=False, prevent_rollback=False, is_training=None):
        """
        forward passing the network of regretful agent

        :param img_feat: (batch_size, 36, d-dim feat)
        :param navigable_feat: (batch_size, max number of navigable direction, d-dim)
        :param pre_feat: (batch_size, d-dim)
        :param pre_value: (batch_size, 1)
        :param h_0: (batch_size, d-dim)
        :param c_0: (batch_size, d-dim)
        :param ctx: (batch_size, max instruction length, d-dim)
        :param navigable_index: list of list, index for navigable directions for each sample in the mini-batch
        :param navigable_idx_to_previous: list
        :param oscillation_index: list
        :param block_oscillation_index: list
        :param ctx_mask: (batch_size, max instruction length)
        :param seq_lengths: list
        :param prevent_oscillation: 1 or 0
        :param prevent_rollback: 1 or 0
        :param is_training: True or False
        """

        batch_size, num_imgs, feat_dim = img_feat.size()

        # creating a mask to block out non-navigable directions, due to batch processing
        index_length = [len(_index) + self.num_predefined_action for _index in navigable_index]
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)

        # prevent rollback action as a sanity check. See Table 3 in the paper.
        if prevent_rollback and not is_training:
            navigable_mask[torch.LongTensor(range(batch_size)), np.array(oscillation_index) + 1] = \
                navigable_mask[torch.LongTensor(range(batch_size)), np.array(oscillation_index) + 1] * (
                            1 - (torch.Tensor(index_length) > 2)).float().to(self.device)

        # block the navigable direction that leads to oscillation
        if 1 in block_oscillation_index and prevent_oscillation:
            navigable_mask = self.block_oscillation(batch_size, navigable_mask, oscillation_index,
                                                    block_oscillation_index)

        # get navigable features without attached markers for visual grounding
        navigable_feat_no_visited = navigable_feat[:, :, :-self.opts.tiled_len]
        proj_navigable_feat = proj_masking(navigable_feat_no_visited, self.proj_navigable_mlp, navigable_mask)
        proj_pre_feat = self.proj_navigable_mlp(pre_feat[:, :-self.opts.tiled_len])
        weighted_img_feat, img_attn = self.soft_attn(self.h0_fc(h_0), proj_navigable_feat, mask=navigable_mask)

        # positional encoding instruction embeddings and textual grounding
        positioned_ctx = self.positional_encoding(ctx)
        weighted_ctx, ctx_attn = self.soft_attn(self.h1_fc(h_0), positioned_ctx, mask=ctx_mask)

        # merge info into one LSTM to be carry through time
        concat_input = torch.cat((proj_pre_feat, weighted_img_feat, weighted_ctx), 1)
        h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
        h_1_drop = self.dropout(h_1)

        # =========== forward and rollback embeddings ===========
        m_forward = self.logit_fc(torch.cat((weighted_ctx, h_1_drop), dim=1))
        m_rollback = proj_navigable_feat[torch.LongTensor(range(batch_size)),
                     np.array(navigable_idx_to_previous) + 1, :]

        # =========== Progress Monitor ===========
        concat_value_input = self.h2_fc_lstm(torch.cat((h_0, weighted_img_feat), 1))
        h_1_value = self.dropout(torch.sigmoid(concat_value_input) * torch.tanh(c_1))
        critics_input = torch.cat((ctx_attn, h_1_value), dim=1)

        if self.opts.monitor_sigmoid:
            value = self.sigmoid(self.critic_fc(critics_input))
        else:
            value = self.tanh(self.critic_fc(critics_input))

        # =========== Progress Marker ===========
        value_detached = value.detach()
        value_for_marker = value_detached.unsqueeze(1).repeat(1, self.max_navigable, self.opts.tiled_len) * \
                           navigable_mask.unsqueeze(2).repeat(1, 1, self.opts.tiled_len)

        navigable_visited_feat = value_for_marker - navigable_feat[:, :, -self.opts.tiled_len:]

        proj_navigable_feat_visited = torch.cat((proj_navigable_feat, navigable_visited_feat), dim=2)

        # =========== Regret Module ===========
        rollback_forward = torch.cat((m_rollback.unsqueeze(1), m_forward.unsqueeze(1)), dim=1)
        rollback_forward_logit = self.critic_valueDiff_fc(value_detached - pre_value)
        rollback_forward_attn = self.softmax(rollback_forward_logit)
        m_forward_rollback = torch.bmm(rollback_forward_attn.unsqueeze(1), rollback_forward).squeeze(1)

        # =========== Action selection with Progress Marker ===========
        logit = torch.bmm(proj_navigable_feat_visited, self.move_fc(m_forward_rollback).unsqueeze(2)).squeeze(2)


        return h_1, c_1, img_attn, ctx_attn, rollback_forward_attn, logit, rollback_forward_logit, value, navigable_mask


class SelfMonitoring(nn.Module):
    """
    The model for the self-monitoring agent (ICLR 2019)

    Self-Monitoring Navigation Agent via Auxiliary Progress Estimation
    arXiv: https://arxiv.org/abs/1901.03035
    GitHub: https://github.com/chihyaoma/selfmonitoring-agent
    Project: https://chihyaoma.github.io/project/2018/09/27/selfmonitoring.html

    """

    def __init__(self, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, rnn_dropout, max_len, fc_bias=True, max_navigable=16):
        super(SelfMonitoring, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_navigable = max_navigable

        proj_navigable_kwargs = {
            'input_dim': img_feat_input_dim,
            'hidden_dims': img_fc_dim,
            'use_batchnorm': img_fc_use_batchnorm,
            'dropout': img_dropout,
            'fc_bias': fc_bias,
            'relu': opts.mlp_relu
        }
        self.proj_navigable_mlp = build_mlp(**proj_navigable_kwargs)

        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)
        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=fc_bias)

        self.soft_attn = SoftAttention()

        self.dropout = nn.Dropout(p=rnn_dropout)

        self.lstm = nn.LSTMCell(img_fc_dim[-1] * 2 + rnn_hidden_size, rnn_hidden_size)

        self.lang_position = PositionalEncoding(rnn_hidden_size, dropout=0.1, max_len=max_len)

        self.logit_fc = nn.Linear(rnn_hidden_size * 2, img_fc_dim[-1])
        self.h2_fc_lstm = nn.Linear(rnn_hidden_size + img_fc_dim[-1], rnn_hidden_size, bias=fc_bias)

        if opts.monitor_sigmoid:
            self.critic = nn.Sequential(
                nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            self.critic = nn.Sequential(
                nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Tanh()
            )

        self.num_predefined_action = 1

    def forward(self, img_feat, navigable_feat, pre_feat, question, h_0, c_0, ctx, pre_ctx_attend,
                navigable_index=None, ctx_mask=None):
        """ Takes a single step in the decoder

        img_feat: batch x 36 x feature_size
        navigable_feat: batch x max_navigable x feature_size
        pre_feat: previous attended feature, batch x feature_size
        question: this should be a single vector representing instruction
        ctx: batch x seq_len x dim
        navigable_index: list of list
        ctx_mask: batch x seq_len - indices to be masked
        """
        batch_size, num_imgs, feat_dim = img_feat.size()

        index_length = [len(_index) + self.num_predefined_action for _index in navigable_index]
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)

        proj_navigable_feat = proj_masking(navigable_feat, self.proj_navigable_mlp, navigable_mask)
        proj_pre_feat = self.proj_navigable_mlp(pre_feat)
        positioned_ctx = self.lang_position(ctx)

        weighted_ctx, ctx_attn = self.soft_attn(self.h1_fc(h_0), positioned_ctx, mask=ctx_mask)

        weighted_img_feat, img_attn = self.soft_attn(self.h0_fc(h_0), proj_navigable_feat, mask=navigable_mask)

        # merge info into one LSTM to be carry through time
        concat_input = torch.cat((proj_pre_feat, weighted_img_feat, weighted_ctx), 1)

        h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
        h_1_drop = self.dropout(h_1)

        # policy network
        h_tilde = self.logit_fc(torch.cat((weighted_ctx, h_1_drop), dim=1))
        logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)

        # value estimation
        concat_value_input = self.h2_fc_lstm(torch.cat((h_0, weighted_img_feat), 1))

        h_1_value = self.dropout(torch.sigmoid(concat_value_input) * torch.tanh(c_1))

        value = self.critic(torch.cat((ctx_attn, h_1_value), dim=1))

        return h_1, c_1, weighted_ctx, img_attn, ctx_attn, logit, value, navigable_mask

class SpeakerFollowerBaseline(nn.Module):
    """
    The baseline model for the speaker-follower (NeurIPS 2018).
    Note that this implementation is only for the basic speaker-follow without Pragmatic Inference

    arXiv: https://arxiv.org/abs/1806.02724
    """

    def __init__(self, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, rnn_dropout, max_len, fc_bias=True, max_navigable=16):
        super(SpeakerFollowerBaseline, self).__init__()

        self.max_navigable = max_navigable

        self.proj_img_mlp = nn.Linear(img_feat_input_dim, img_fc_dim[-1], bias=fc_bias)

        self.proj_navigable_mlp = nn.Linear(img_feat_input_dim, img_fc_dim[-1], bias=fc_bias)

        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=False)

        self.soft_attn = SoftAttention()

        self.dropout = nn.Dropout(p=rnn_dropout)

        self.lstm = nn.LSTMCell(img_feat_input_dim * 2, rnn_hidden_size)

        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=False)

        self.proj_out = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)

    def forward(self, img_feat, navigable_feat, pre_feat, h_0, c_0, ctx, navigable_index=None, ctx_mask=None):
        """ Takes a single step in the decoder LSTM.

        img_feat: batch x 36 x feature_size
        navigable_feat: batch x max_navigable x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        navigable_index: list of list
        ctx_mask: batch x seq_len - indices to be masked
        """
        batch_size, num_imgs, feat_dim = img_feat.size()

        # add 1 because the navigable index yet count in "stay" location
        # but navigable feature does include the "stay" location at [:,0,:]
        index_length = [len(_index)+1 for _index in navigable_index]
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)

        proj_img_feat = proj_masking(img_feat, self.proj_img_mlp)

        proj_navigable_feat = proj_masking(navigable_feat, self.proj_navigable_mlp, navigable_mask)

        weighted_img_feat, _ = self.soft_attn(self.h0_fc(h_0), proj_img_feat, img_feat)

        concat_input = torch.cat((pre_feat, weighted_img_feat), 1)

        h_1, c_1 = self.lstm(self.dropout(concat_input), (h_0, c_0))

        h_1_drop = self.dropout(h_1)

        # use attention on language instruction
        weighted_context, ctx_attn = self.soft_attn(self.h1_fc(h_1_drop), self.dropout(ctx), mask=ctx_mask)
        h_tilde = self.proj_out(weighted_context)

        logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)

        return h_1, c_1, ctx_attn, logit, navigable_mask