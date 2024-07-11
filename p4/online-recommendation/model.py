import mindspore as ms
import networkx as nx
import mindspore as ms
from mindspore import nn
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from queue import *
from random import sample
import math

from typing import List
import mindspore as ms
from mindspore_gl.nn.conv import GATConv
from mindspore_gl import GNNCell
from mindspore_gl import HeterGraph

class MetagraphAttention(ms.nn.Cell):
    def __init__(self,
                 in_feat_size: int,
                 hidden_size: int = 128) -> None:
        super().__init__()
        self.proj = ms.nn.SequentialCell(
            ms.nn.Dense(in_feat_size, hidden_size),
            ms.nn.Tanh(),
            ms.nn.Dense(hidden_size, 1, has_bias=False)
        )

    def construct(self, x):
        """construct function"""
        h = ms.ops.ReduceMean()(self.proj(x), 0)
        beta = ms.ops.Softmax(0)(h)
        beta = ms.ops.BroadcastTo((ms.ops.Shape()(x)[0],) + ms.ops.Shape()(beta))(beta)
        return ms.ops.ReduceSum()(beta * x, 1)

class MetagraphLayer(GNNCell):
    def __init__(self,
                 num_meta_paths: int,
                 in_feat_size: int,
                 out_size: int,
                 num_heads: int,
                 dropout: float) -> None:
        super().__init__()
        gats = []
        print("in_feat size", in_feat_size)
        for _ in range(num_meta_paths):
            gats.append(GATConv(in_feat_size, out_size, num_heads, dropout, dropout, activation=ms.nn.ELU()))

        self.gats = ms.nn.CellList(gats)
        self.semantic = MetagraphAttention(out_size * num_heads)
        self.num_meta_paths = num_meta_paths

    def construct(self, h, hg: HeterGraph):
        """construct function"""
        metagraph_embeddings = []
        for i in range(self.num_meta_paths):
            metagraph_embeddings.append(self.gats[i](h, *hg.get_homo_graph(i)))

        metagraph_embeddings = ms.ops.Stack(1)(metagraph_embeddings)
        ret = self.semantic(metagraph_embeddings)
        return ret

class LossNet(ms.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss_fn = ms.nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    def construct(self, h, target, train_idx, src_idx, dst_idx, n_nodes, n_edges):
        pred = self.net(h, src_idx, dst_idx, n_nodes, n_edges)
        loss = self.loss_fn(pred[train_idx], target)
        return loss



class HAN(GNNCell):
    """HAN"""
    def __init__(self,
                 num_meta_paths: int,
                 in_feat_size: int,
                 hidden_size: int,
                 out_size: int,
                 num_heads: List[int],
                 dropout: float
                 ) -> None:
        super().__init__()
        layers = [HANLayer(num_meta_paths, in_feat_size, hidden_size, num_heads[0], dropout)]
        for i in range(1, len(num_heads)):
            layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[i - 1], hidden_size, num_heads[i], dropout))
        self.layers = ms.nn.CellList(layers)
        self.predict = ms.nn.Dense(hidden_size * num_heads[-1], out_size)

    def construct(self, h, hg: HeterGraph):
        """construct function"""
        for conv in self.layers:
            h = conv(h, hg)
        return self.predict(h)


class DHANE_Model:
    
    def __init__(self, data, args):
        self.data = data
        self.args = args
        self.d = args["d"]
        self.h = args["h"]
        self.he = args["he"]
        self.ha = args["ha"]
        self.r = args["r"]
        self.re = args["re"]
        self.ra = args["ra"]
        self.relu_alpha = args["relu_alpha"]
        self.metapath_cutoff = args["metapath_cutoff"]
        self.node_embedding = []
        self.graph = data.nx_net
        self.pre_train_edge_weights = [[1] * self.data.node_types] * self.data.node_types # '1' is only for initialize, actual element is a MxN matrix

    def attention_coefficient(x, y, y_all, theta="i"):
        t1 = np.dot(theta, (x + y))
        t1 = t1 if t1 > 0 else t1 * self.relu_alpha
        t1 = np.exp(t1)

        t3 = 0
        for y2 in x_all:
            t2 = np.dot(theta, (x + y2))
            t2 = t2 if t2 > 0 else t2 * self.relu_alpha
            t3 += np.exp(t2)

        return t1/t3

    def pre_train_edge_weights(self):
        print("begin pre train edge weights")

        for i in range(self.data.node_types):
            for j in range(self.data.node_types):
                n = self.data.type_nodes_feature_cnt[i]
                m = self.data.type_nodes_feature_cnt[j]
                array1 = self.data.type_nodes_feature_matrix[i]
                array2 = self.data.type_nodes_feature_matrix[j]
                self.pre_train_edge_weights[i][j] = np.ones((n, m))
                for feature1 in range(n):
                    x = array1[:, feature1]
                    y_all = [array2[:, feature2] for feature2 in range(m)]
                    for feature2 in range(m):
                        self.pre_train_edge_weights[i][j][feature1, feature2] = attention_coefficient(x, y_all[feature2], y_all)

        print(self.pre_train_edge_weights)
        print("finish pre train edge weights")

        
    def find_neighbours_in_h_hoop(self, v):
        q = Queue()
        q.put((v[0],0))
        visited = set()
        visited.add(v[0])
        while not q.empty():
            v, distance = q.get()
            n_list = nx.neighbors(self.graph, v)
            for node in n_list:
                if node not in visited and distance < self.h:
                    q.put((node, distance+1))
                    visited.add(node)
        visited.remove(v)
        print(v, len(visited))
        return visited

    def metagraph_generation(self, v):
        neighbors = self.find_neighbours_in_h_hoop(v)
        #find all neighbours in h hoop
        metagraph_dic_v = {}
        metapaths_dic_v = {}
        for n in neighbors:
            metagraph_dic_v[n] = []
            print(n, v)
            metapaths_dic_v[n] = sorted(nx.all_simple_edge_paths(self.graph, v[0], n, cutoff=self.metapath_cutoff))
            print("all simple path", len(metapaths_dic_v[n]))
            for path in sorted(nx.all_simple_edge_paths(self.graph, v, n)):
                for edge in path:
                    metagraph_dic_v[n].append(edge)

        return neighbors, metapaths_dic_v

    def get_mgat_embedding(self):
        for v in self.data.nodes:
            neighbors, meta_paths = self.metagraph_generation(v)
            sample_neighbors = sample(neighbors, math.ceil(len(neighbors) * self.r))
            for n in sample_neighbors:
                metagraph_info = self.metagraph_info_computation(v, n, meta_paths[n])
            self.metagraph_info_aggregation(v)
            self.node_embedding.append(self.static_embedding_output(v))


    def dynamic_embedding_update(self, t):
        delta = self.edges[t]
        for t in range(t):
            delta_edge = self.data.time_edges[t-1]


    def train(self):
        print("start training")
        optimizer = nn.optim.Adam(net.trainable_params(), learning_rate=arguments.lr, weight_decay=arguments.weight_decay)
        loss = LossNet(net)
        train_net = nn.TrainOneStepCell(loss, optimizer)
        warm_up = 3
        total = 0.
        for e in range(arguments.epochs):
            beg = time.time()
            train_net.set_train()
            train_loss = train_net(features, train_labels, train_idx, *hgf.get_heter_graph())

            end = time.time()
            dur = end - beg
            if e >= warm_up:
                total = total + dur

            net.set_train(False)
            out = net(features, *hgf.get_heter_graph())
            test_predict = out[test_idx].asnumpy().argmax(axis=1)
            test_label = labels[test_idx].asnumpy()
            count = np.equal(test_predict, test_label)
            print('Epoch:{} Epoch time:{} ms Train loss {} Test acc:{}'.format(e, dur * 1000, train_loss,
                                                                            np.sum(count) / test_label.shape[0]))
            print("Model:{} Dataset:{} Avg epoch time:{}".format("HAN", arguments.data_path,
                                                            total * 1000 / (arguments.epochs - warm_up)))


    def test(self, test_classifier, test_data, test_label):
        for classifier in test_classifier:
            if classifier == "random_forest":
                random_forest_classifier(test_data, test_label)


    def random_forest_classifier(self, x_train, y_train, x_test, y_test):
        rfc = RandomForestClassifier(random_state=0, n_jobs=2)
        rfc = rfc.fit(x_train, y_train)
        score = rfc.score(x_test, y_test)
