# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import mindspore
import mindspore.nn as nn
from .builder import MODELS


__all__ = ['resnet18_cifar_nce', 'resnet34_cifar_nce', 'resnet50_cifar_nce',
           'resnet101_cifar_nce', 'resnet152_cifar_nce']

def Normalize(feat):
    l2_normalize = mindspore.ops.L2Normalize(axis = 1)
    return l2_normalize(feat.float())

def chord_distance_func(feat1, feat2):
    # Chord Distance @
    # feat1: N * Dim @
    # feat2: M * Dim @
    # out:   N * M Chord Distance @
    feat1, feat2 = Normalize(feat1), Normalize(feat2)
    feat_matmul = mindspore.ops.matmul(feat1, feat2.t())
    distance = mindspore.ops.ones_like(feat_matmul) - feat_matmul
    distance = distance * 2
    return distance.clamp(1e-10).sqrt()


def euclidean_distance_func(feat1, feat2):
    # Euclidean Distance
    # feat1: N * Dim
    # feat2: M * Dim
    # out:   N * M Euclidean Distance
    feat1_square = mindspore.ops.sum(mindspore.ops.pow(feat1, 2), 1, keepdim=True)
    feat2_square = mindspore.ops.sum(mindspore.ops.pow(feat2, 2), 1, keepdim=True)
    feat_matmul = mindspore.ops.matmul(feat1, feat2.t())
    distance = feat1_square + feat2_square.t() - 2 * feat_matmul
    return distance.clamp(1e-10).sqrt()


def cosine_distance_func(feat1, feat2):
    # feat1: N * Dim @
    # feat2: M * Dim @
    # out:   N * M Cosine Distance @
    feat1, feat2 =Normalize(feat1), Normalize(feat2)
    distance = mindspore.ops.matmul(feat1, feat2.t())

    return distance


def cosine_distance_full_func(feat1, feat2):
    # feat1: N * Dim
    # feat2: M * Dim
    # out:   (N+M) * (N+M) Cosine Distance

    feat = mindspore.ops.cat((feat1, feat2), axis=0)
    distance = mindspore.ops.matmul(Normalize(feat), Normalize(feat).t())
    return distance


# class BasicBlock(nn.Module):
class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, pad_mode='pad',has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, pad_mode='pad',has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.SequentialCell(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, has_bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
            )

    def construct(self, x):
        out = mindspore.ops.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return mindspore.ops.relu(out)

class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, pad_mode='pad', has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.SequentialCell(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, has_bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
            )

    def construct(self, x):
        out = mindspore.ops.relu(self.bn1(self.conv1(x)))
        out = mindspore.ops.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return mindspore.ops.relu(out)


# class ResNet(nn.Module):
class ResNet(nn.Cell):
    def __init__(self, block, num_blocks, num_classes=10, alpha=0.5, num_proto=1, drop_ratio=0.):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.alpha = alpha
        self.num_proto   = num_proto
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad',has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv5  = nn.Conv2d(512, 512, kernel_size=1, has_bias=False)
        self.dropout = mindspore.nn.Dropout(p=drop_ratio)

    
        self.protos = mindspore.Parameter(
            mindspore.ops.randn(self.num_proto * self.num_classes, 512),
            requires_grad=True
        )

        self.radius = mindspore.Parameter(
            mindspore.ops.rand(1, self.num_proto * self.num_classes) - 0.5,
            requires_grad=True
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.SequentialCell(*layers)
    

    def nce_prob_cos(self, feat):
        dist = cosine_distance_func(feat, self.protos)
        dist = (dist / self.radius.sigmoid()).sigmoid()
        cls_score, _ = dist.view(-1, self.num_proto, self.num_classes).max(axis=1, return_indices=True)
        return cls_score

    def nce_prob_euc(self, feat):
        dist = chord_distance_func(feat.sigmoid(), self.protos.sigmoid())
        cls_score, _ = dist.view(-1, self.num_proto, self.num_classes).max(axis=1, return_indices=True)
        cls_score = mindspore.ops.exp(-(cls_score ** 2) / (2 * self.radius.sigmoid() ** 2))
        return cls_score

    def construct(self, x):
        feat =  mindspore.ops.relu(self.bn1(self.conv1(x)))
        feat1 = self.layer1(feat)
        feat1 = self.dropout(feat1)
        feat2 = self.layer2(feat1)
        feat2 = self.dropout(feat2)
        feat3 = self.layer3(feat2)
        feat3 = self.dropout(feat3)
        feat4 = self.layer4(feat3)
        feat4 = self.dropout(feat4)
        outf = self.conv5(mindspore.ops.avg_pool2d(feat4, 4)).view(feat4.shape[0], -1)
        prob = self.nce_prob_euc(outf)
        return prob, outf, [feat1, feat2, feat3, feat4]


@MODELS.register_module()
class resnet18_cifar_nce(object):
    def __init__(self):
        pass

    def __call__(self, **kwargs) -> ResNet:
        return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


@MODELS.register_module()
class resnet34_cifar_nce(object):
    def __init__(self):
        pass

    def __call__(self, **kwargs) -> ResNet:
        return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


@MODELS.register_module()
class resnet50_cifar_nce(object):
    def __init__(self):
        pass

    def __call__(self, **kwargs) -> ResNet:
        return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


@MODELS.register_module()
class resnet101_cifar_nce(object):
    def __init__(self):
        pass

    def __call__(self, **kwargs) -> ResNet:
        return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


@MODELS.register_module()
class resnet152_cifar_nce(object):
    def __init__(self):
        pass

    def __call__(self, **kwargs) -> ResNet:
        return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
