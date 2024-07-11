import math
import mindspore as ms
import mindspore.nn as nn


class GraphConvolution(nn.Cell):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = ms.Parameter(ms.Tensor(shape = (in_features, out_features), dtype=ms.float32))
        if bias:
            self.bias = ms.Parameter(ms.Tensor(shape = out_features, dtype=ms.float32))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        uniform_init = ms.common.initializer.Uniform(-stdv,stdv)
        uniform_init(self.weight.default_input)

        if self.bias is not None:
            uniform_init = ms.common.initializer.Uniform(-stdv,stdv)
            uniform_init(self.bias.default_input)

    def construct(self, input, adj):
        support = ms.ops.mm(input, self.weight)
        output = ms.ops.matmul(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Cell):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear = nn.Dense(nclass, 1)

    def construct(self, x, adj):
        x = ms.ops.relu(self.gc1(x, adj))
        feat = ms.ops.dropout(x, self.dropout, training=self.training)
        x = self.gc3(feat, adj)
        return ms.ops.sigmoid(x), feat, ms.ops.cat((feat, x), 1)
