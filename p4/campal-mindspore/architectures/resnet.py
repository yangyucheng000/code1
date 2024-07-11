# import torch
# from torch import Tensor
# import torch.nn as nn
import math
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from .builder import MODELS


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation,pad_mode='pad',group=groups, has_bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, has_bias=False)


class BasicBlock(nn.Cell):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Cell] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Cell]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Cell] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Cell]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Cell):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        drop_ratio=0.0,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Cell]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, pad_mode='pad', has_bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,pad_mode='pad')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Dense(512 * block.expansion, num_classes)
        self.dropout = nn.Dropout(p=drop_ratio)


        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(mindspore.common.initializer.initializer(
                    mindspore.common.initializer.HeNormal(negative_slope=0, mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
            elif isinstance(cell, (nn.BatchNorm2d, nn.GroupNorm)):
                cell.gamma.set_data(mindspore.common.initializer.initializer("ones", cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(mindspore.common.initializer.initializer("zeros", cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, (nn.Dense)):
                cell.weight.set_data(mindspore.common.initializer.initializer(
                    mindspore.common.initializer.HeUniform(negative_slope=math.sqrt(5)),
                    cell.weight.shape, cell.weight.dtype))
                cell.bias.set_data(mindspore.common.initializer.initializer("zeros", cell.bias.shape, cell.bias.dtype))

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for _, cell in self.cells_and_names():
                if isinstance(cell, Bottleneck) and cell.bn3.gamma is not None:
                    cell.bn3.gamma.set_data("zeros", cell.bn3.gamma.shape, cell.bn3.gamma.dtype)
                elif isinstance(cell, BasicBlock) and cell.bn2.weight is not None:
                    cell.bn2.gamma.set_data("zeros", cell.bn2.gamma.shape, cell.bn2.gamma.dtype)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.SequentialCell:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.SequentialCell(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out1 = self.layer1(x)
        out1 = self.dropout(out1)
        out2 = self.layer2(out1)
        out2 = self.dropout(out2)
        out3 = self.layer3(out2)
        out3 = self.dropout(out3)
        out4 = self.layer4(out3)
        out4 = self.dropout(out4)

        outf = self.avgpool(out4)
        outf = mindspore.ops.flatten(outf, 1)
        out = self.fc(outf)

        return out, outf, [out1, out2, out3, out4]

    def construct(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


@MODELS.register_module()
class resnet18(object):
    """ResNet-18 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

        Args:
            pretrained: (bool)
                If True, returns a model pre-trained on ImageNet
            progress: (bool)
                If True, displays a progress bar of the download to stderr

    """
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        """Return the model architecture. Parameter `num_classes` must be included.

            :param num_classes: (int) The number of classes of the input dataset

        """
        return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class resnet34(object):
    """ResNet-34 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

        Args:
            pretrained: (bool)
                If True, returns a model pre-trained on ImageNet
            progress: (bool)
                If True, displays a progress bar of the download to stderr

    """
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        """Return the model architecture. Parameter `num_classes` must be included.

            :param num_classes: (int) The number of classes of the input dataset

        """
        return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class resnet50(object):
    """ResNet-50 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

        Args:
            pretrained: (bool)
                If True, returns a model pre-trained on ImageNet
            progress: (bool)
                If True, displays a progress bar of the download to stderr

    """
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        """Return the model architecture. Parameter `num_classes` must be included.

            :param num_classes: (int) The number of classes of the input dataset

        """
        return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class resnet101(object):
    """ResNet-101 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

        Args:
            pretrained: (bool)
                If True, returns a model pre-trained on ImageNet
            progress: (bool)
                If True, displays a progress bar of the download to stderr

    """
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        """Return the model architecture. Parameter `num_classes` must be included.

            :param num_classes: (int) The number of classes of the input dataset

        """
        return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class resnet152(object):
    """ResNet-152 model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

        Args:
            pretrained: (bool)
                If True, returns a model pre-trained on ImageNet
            progress: (bool)
                If True, displays a progress bar of the download to stderr

    """
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        """Return the model architecture. Parameter `num_classes` must be included.

            :param num_classes: (int) The number of classes of the input dataset

        """
        return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class resnext50_32x4d(object):
    """ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

        Args:
            pretrained: (bool)
                If True, returns a model pre-trained on ImageNet
            progress: (bool)
                If True, displays a progress bar of the download to stderr

    """
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        """Return the model architecture. Parameter `num_classes` must be included.

            :param num_classes: (int) The number of classes of the input dataset

        """
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 4
        return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class resnext101_32x8d(object):
    """ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

        Args:
            pretrained: (bool)
                If True, returns a model pre-trained on ImageNet
            progress: (bool)
                If True, displays a progress bar of the download to stderr

    """
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        """Return the model architecture. Parameter `num_classes` must be included.

            :param num_classes: (int) The number of classes of the input dataset

        """
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 8
        return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class wide_resnet50_2(object):
    """Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

        Args:
            pretrained: (bool)
                If True, returns a model pre-trained on ImageNet
            progress: (bool)
                If True, displays a progress bar of the download to stderr

    """
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        """Return the model architecture. Parameter `num_classes` must be included.

            :param num_classes: (int) The number of classes of the input dataset

        """
        kwargs['width_per_group'] = 64 * 2
        return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class wide_resnet101_2(object):
    """Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

        Args:
            pretrained: (bool)
                If True, returns a model pre-trained on ImageNet
            progress: (bool)
                If True, displays a progress bar of the download to stderr

    """
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        """Return the model architecture. Parameter `num_classes` must be included.

            :param num_classes: (int) The number of classes of the input dataset

        """
        kwargs['width_per_group'] = 64 * 2
        return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], self.pretrained, self.progress,
                       **kwargs)
