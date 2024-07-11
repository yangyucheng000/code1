import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore import dtype
import numpy as np
import losses

VOCAB_SIZE = 20000
ld = 1

def triplet_loss(anchor, positive, negative):
    margin = 1
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = ops.relu(distance_positive - distance_negative + margin)
    return losses

"""
class FsAdaptLoss(nn.LossBase):
    def __init__(self, margin):
        self.margin = margin

    def triplet_loss(anchor, positive, )
    
    def construct(self, )
"""



class EmbeddingNet(nn.Cell):
    def __init__(self, input_shape):
        super().__init__()
        self.features = nn.SequentialCell([nn.Conv1d(input_shape, 128, kernel_size=8), 
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     nn.Conv1d(128, 256, kernel_size=5),
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(),
                                     nn.Conv1d(256, 128, kernel_size=3),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(),
                                     nn.AvgPool1d(pad_mode='pad', stride=20),
                                     ])

    def construct(self, x):
        x = x.astype(ms.float32)
        output = self.features(x)
        return output

class TripletNet(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x_anchor, x1, x2, y1, y2):
        if ops.dist(x_anchor, x1) < ops.dist(x_anchor, x2):
            return y1, triplet_loss(x_anchor, x1, x2)
        else:
            return y2, triplet_loss(x_anchor, x1, x2)

class ClassificationNet(nn.Cell):
    def __init__(self, embedding_net, n_classes):
        super().__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def construct(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))

class GradientReversalLayer(nn.Cell):
    def __init__(self):
        super().__init__()
        
    def construct(self, input):
        return input

    def bprop(self, input, output, grad_output):
        return input.neg()

class FsAdaptNet(nn.Cell):
    """
    FsAdapt的整个网络分为3部分
    feature_extractor:
    domain_classifier:
    class_classifier:
    gradient reversal layer
    """
    def __init__(self, input_shape):
        super().__init__()
        self.feature_extractor = EmbeddingNet(input_shape)
        self.domain_classifier = TripletNet()
        self.class_classifier = TripletNet()

    def construct(self, x0, x1, x2, x3, y_class, y_domain):
        x0 = self.feature_extractor(x0)
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        x3 = self.feature_extractor(x3)

        x_domain0, loss0 = self.domain_classifier(x2, x0, x1, y_domain[0][0], y_domain[0][1])
        x_domain1, loss1 = self.domain_classifier(x2, x1, x3, y_domain[0][1], y_domain[0][3])
        x_class, loss2 = self.class_classifier(x2, x1, x3, y_class[0][1], y_class[0][3])
        return x_class, loss2 + ld * (loss0 + loss1)