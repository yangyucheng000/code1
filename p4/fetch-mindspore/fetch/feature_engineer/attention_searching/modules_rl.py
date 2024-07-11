import logging

# import torch
# import torch.nn as nn
# import torch.nn.init as init
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore import ops
from mindspore import Tensor
# from mindspore import Tensor, Parameter, context
from mindspore.common.initializer import initializer as init
from mindspore.common.initializer import XavierNormal, Constant, HeNormal


def weight_init(m):
    # if isinstance(m, nn.Linear):
    if isinstance(m, nn.Dense):
        # nn.init.xavier_normal_(m.weight)
        m.weight = init(XavierNormal())
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.constant_(m.bias, 0)
        m.bias = init(Constant(0))
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv1d):
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        m.weight = init(HeNormal())
     # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm2d):
        # nn.init.constant_(m.weight, 1)
        # nn.init.constant_(m.bias, 0)
        m.weight = init(Constant(1))
        m.bias = init(Constant(0))


class ScaledDotProductAttention(nn.Cell):
    def __init__(self, d_k, dropout=None):
        super(ScaledDotProductAttention, self).__init__()
        # print("scaleddot init begin")
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(axis=-1)
        self.dropout_sign = dropout
        if self.dropout_sign:
            # self.dropout = nn.Dropout(dropout, inplace=True)
            self.dropout = nn.Dropout(1.0 - dropout)
        # print("scaleddot init done")

    # def forward(self, q, k, v, attn_mask=None):
    def construct(self, q, k, v, attn_mask=None):
        # q: [b_size x n_heads x len_q x d_k]
        # k: [b_size x n_heads x len_k x d_k]
        # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

        # attn: [b_size x n_heads x len_q x len_k]
        # scores = mindspore.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        # print("scaleddot construct begin")
        scores = mindspore.ops.matmul(q, k.swapaxes(-1, -2)) / self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)
        if self.dropout_sign:
            attn = self.dropout(self.softmax(scores))
        else:
            attn = self.softmax(scores)

        # outputs: [b_size x n_heads x len_q x d_v]
        context = mindspore.ops.matmul(attn, v)
        # print("scaleddot construct done")
        return context, attn


# class _MultiHeadAttention(nn.Module):
class _MultiHeadAttention(nn.Cell):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout):
        # d_model = 128,d_k=d_v = 32,n_heads = 4
        super(_MultiHeadAttention, self).__init__()
        # print("_multihead init begin")
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        # self.w_q = Linear(d_model, d_k * n_heads)
        # self.w_k = Linear(d_model, d_k * n_heads)
        # self.w_v = Linear(d_model, d_v * n_heads)
        self.w_q = nn.Dense(d_model, d_k * n_heads)
        self.w_k = nn.Dense(d_model, d_k * n_heads)
        self.w_v = nn.Dense(d_model, d_v * n_heads)

        self.attention = ScaledDotProductAttention(d_k, dropout)
        # print("_multihead init done")

    # def forward(self, q, k, v, attn_mask):
    def construct(self, q, k, v, attn_mask):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        # print("_multihead construct begin")
        # print(type(q))
        # print(type(q.shape))
        # b_size = q.size(0)
        b_size = q.shape[0]

        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]
        # q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_k).swapaxes(1, 2)
        k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_k).swapaxes(1, 2)
        v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_v).swapaxes(1, 2)
        # print("@@@")

        if attn_mask:  # attn_mask: [b_size x len_q x len_k]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # context: [b_size x n_heads x len_q x d_v], attn: [b_size x n_heads x len_q x len_k]
        context, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)
        # context: [b_size x len_q x n_heads * d_v]
        # context = context.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)
        # print("@@@")
        
        context_shape = context.shape

        transpose_op = ops.Transpose()
        context = transpose_op(context, (0, 2, 1, 3))
        context = ops.reshape(context, (context_shape[0], -1, self.n_heads * self.d_v))
        
        # print(type(context))
        # context = context.swapaxes(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)
        
        # print("_multihead construct done")
        # return the context and attention weights
        return context, attn


# class MultiHeadAttention(nn.Module):
class MultiHeadAttention(nn.Cell):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout=None):
        super(MultiHeadAttention, self).__init__()
        # print("multihead init begin")
        self.n_heads = n_heads
        self.multihead_attn = _MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        # self.proj = nn.Linear(n_heads * d_v, d_model)
        self.proj = nn.Dense(n_heads * d_v, d_model)
        # self.layer_norm = LayerNormalization(d_model)
        self.layer_norm = nn.LayerNorm((d_model, ))
        self.dropout_sign = dropout
        if self.dropout_sign:
            # self.dropout = nn.Dropout(dropout, inplace=True)
            self.dropout = nn.Dropout(1.0 - dropout)
        # print("multihead init done")

    # def forward(self, q, k, v, attn_mask=None):
    def construct(self, q, k, v, attn_mask=None):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        # print("multihead construct begin")
        residual = q
        # context: a tensor of shape [b_size x len_q x n_heads * d_v]
        context, attn = self.multihead_attn(q, k, v, attn_mask=attn_mask)

        # context = mindspore.where(mindspore.ops.isnan(context), mindspore.ops.full_like(context, 0), context)
        # attn = mindspore.where(mindspore.ops.isnan(attn), mindspore.ops.full_like(attn, 0), attn)
        
        isnan_context = ops.IsNan()(context)
        context = ops.select(isnan_context, ops.ZerosLike()(context), context)
        isnan_attn = ops.IsNan()(attn)
        attn = ops.select(isnan_attn, ops.ZerosLike()(attn), attn)

        # project back to the residual size, outputs: [b_size x len_q x d_model]
        if self.dropout_sign:
            output = self.dropout(self.proj(context))
        else:
            output = self.proj(context)

        ro = residual + output
        no = self.layer_norm(ro)
        # print("multihead construct done")
        if mindspore.ops.isnan(no).any() and not mindspore.ops.isnan(ro).any():
            return ro, attn
        return no, attn


# class PoswiseFeedForwardNet(nn.Module):
class PoswiseFeedForwardNet(nn.Cell):
    def __init__(self, d_model, d_ff, dropout=None):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, has_bias=True, pad_mode='valid')
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, has_bias=True, pad_mode='valid')
        self.layer_norm = nn.LayerNorm((d_model, ))
        self.dropout_sign = dropout
        if self.dropout_sign:
            # self.dropout = nn.Dropout(dropout, inplace=True)
            self.dropout = nn.Dropout(1.0 - dropout)

    # def forward(self, inputs):
    def construct(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        # output = self.relu(self.conv1(inputs.transpose(1, 2)))
        output = self.relu(self.conv1(inputs.swapaxes(1, 2)))

        # outputs: [b_size x len_q x d_model]
        # output = self.conv2(output).transpose(1, 2)
        output = self.conv2(output).swapaxes(1, 2)
        if self.dropout_sign:
            output = self.dropout(output)

        return self.layer_norm(residual + output)


# class EncoderLayer(nn.Module):
class EncoderLayer(nn.Cell):
    def __init__(self, d_model, d_k, d_v, d_ff, n_heads, dropout=None):
        # print("encodelayer init begin")
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        # print("encodelayer init done")

    # def forward(self, enc_inputs, self_attn_mask=None):
    def construct(self, enc_inputs, self_attn_mask=None):
        # print("encodelayer construct begin")
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, attn_mask=self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        # print("encodelayer construct done")
        return enc_outputs


# class SelectOperations(nn.Module):
class SelectOperations(nn.Cell):
    def __init__(self, d_model, operations):
        super(SelectOperations, self).__init__()
        self.selector = nn.SequentialCell(
            nn.Dense(d_model, operations),
        )

    # def forward(self, enc_outputs):
    def construct(self, enc_outputs):
        x = enc_outputs.squeeze()[0:-1]
        output = self.selector(x)
        # out = torch.softmax(output, dim=-1)
        return output


# class StatisticLearning(nn.Module):
class StatisticLearning(nn.Cell):
    def __init__(self, statistic_nums, d_model):
        super(StatisticLearning, self).__init__()
        # self.layer = nn.Sequential(
        #     nn.Linear(statistic_nums, statistic_nums * 2),
        #     nn.ReLU(),
        #     nn.Linear(statistic_nums * 2, d_model),
        # )
        self.layer = nn.SequentialCell(
            nn.Dense(statistic_nums, statistic_nums * 2),
            nn.ReLU(),
            nn.Dense(statistic_nums * 2, d_model),
        )

    # def forward(self, input):
    def construct(self, input):
        return self.layer(input)


# class ReductionDimension(nn.Module):
class ReductionDimension(nn.Cell):
    def __init__(self, statistic_nums, d_model):
        super(ReductionDimension, self).__init__()
        # self.layer = nn.Sequential(
        #     nn.BatchNorm1d(statistic_nums),
        #     nn.Linear(statistic_nums, d_model),
        #     nn.BatchNorm1d(d_model),
        # )
        # print("-------------")
        # print(statistic_nums)
        # print(d_model)
        # print("-------------")
        self.layer = nn.SequentialCell(
            nn.BatchNorm1d(statistic_nums),
            nn.Dense(statistic_nums, d_model),
            nn.BatchNorm1d(d_model),
        )

    # def forward(self, input):
    def construct(self, input):
        out = self.layer(input).unsqueeze(dim=0)
        return out


