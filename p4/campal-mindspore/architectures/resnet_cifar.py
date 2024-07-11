#!coding:utf-8

import mindspore
import mindspore.nn as nn
from .builder import MODELS


__all__ = ['resnet18_cifar', 'resnet34_cifar', 'resnet50_cifar',
           'resnet101_cifar', 'resnet152_cifar']


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, pad_mode='pad',  has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False)
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


class ResNet(nn.Cell):
    
    def __init__(self, block, num_blocks, num_classes, drop_ratio=0.0):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc1 = nn.Dense(512*block.expansion, num_classes)
        self.dropout = mindspore.nn.Dropout(p=drop_ratio)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.SequentialCell(*layers)


    def construct(self, x):
        out = mindspore.ops.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.dropout(out1)
        out2 = self.layer2(out2)
        out3 = self.dropout(out2)
        out3 = self.layer3(out3)
        out4 = self.dropout(out3)
        out4 = self.layer4(out4)
        outf = self.dropout(out4)
        outf = mindspore.ops.avg_pool2d(outf, 4)
        outf = outf.view(outf.shape[0], -1)
        return self.fc1(outf), outf, [out1, out2, out3, out4]
    

    def disable_dropout(self, layer: nn.Cell):
        if type(layer) == nn.Dropout:
            layer.set_train(False)


    def disable_all_dropout(self):
        self.apply(self.disable_dropout)


@MODELS.register_module()
class resnet18_cifar(object):
    def __init__(self):
        pass

    def __call__(self, **kwargs) -> ResNet:
        """Return the model architecture. Parameter `num_classes` must be included.

            :param num_classes: (int) The number of classes of the input dataset

        """
        return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


@MODELS.register_module()
class resnet34_cifar(object):
    def __init__(self):
        pass

    def __call__(self, **kwargs) -> ResNet:
        """Return the model architecture. Parameter `num_classes` must be included.

            :param num_classes: (int) The number of classes of the input dataset

        """
        return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


@MODELS.register_module()
class resnet50_cifar(object):
    def __init__(self):
        pass

    def __call__(self, **kwargs) -> ResNet:
        """Return the model architecture. Parameter `num_classes` must be included.

            :param num_classes: (int) The number of classes of the input dataset

        """
        return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


@MODELS.register_module()
class resnet101_cifar(object):
    def __init__(self):
        pass

    def __call__(self, **kwargs) -> ResNet:
        """Return the model architecture. Parameter `num_classes` must be included.

            :param num_classes: (int) The number of classes of the input dataset

        """
        return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


@MODELS.register_module()
class resnet152_cifar(object):
    def __init__(self):
        pass

    def __call__(self, **kwargs) -> ResNet:
        """Return the model architecture. Parameter `num_classes` must be included.

            :param num_classes: (int) The number of classes of the input dataset

        """
        return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == '__main__':
    print('--- run resnet test ---')
    # x = torch.randn(2, 3, 32, 32)
    x = mindspore.ops.randn(2, 3, 32, 32)
    for net in [resnet18_cifar()(num_classes=10), resnet34_cifar()(num_classes=10), resnet50_cifar()(num_classes=10),
                resnet101_cifar()(num_classes=10), resnet152_cifar()(num_classes=10)]:
        print(net)
        y = net(x)
        print(y.shape)
