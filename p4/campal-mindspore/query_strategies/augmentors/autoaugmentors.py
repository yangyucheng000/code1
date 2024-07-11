from enum import Enum
from typing import List, Tuple, Optional, Dict
from .operators_single import apply_op

import mindspore
import numpy as np
from mindspore import nn, Tensor

__all__ = ["AutoAugmentPolicy", "AutoAugment", "RandAugment", "TrivialAugmentWide"]


# 本节所有模块的核心在于对每张图像，能返回一个增强函数
# 注意不要直接返回增强图像


class AutoAugmentPolicy(Enum):
    """AutoAugment policies learned on different datasets.
    Available policies are IMAGENET, CIFAR10 and SVHN.
    """
    IMAGENET = "imagenet"
    CIFAR10 = "cifar10"
    SVHN = "svhn"


# FIXME: Eliminate copy-pasted code for fill standardization and _augmentation_space() by moving stuff on a base class
class AutoAugment(nn.Cell):
    r"""AutoAugment data augmentation method based on
    `"AutoAugment: Learning Augmentation Strategies from Data" <https://arxiv.org/pdf/1805.09501.pdf>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        policy (AutoAugmentPolicy): Desired policy enum defined by
            :class:`torchvision.transforms.autoaugment.AutoAugmentPolicy`. Default is ``AutoAugmentPolicy.IMAGENET``.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        policy: AutoAugmentPolicy = AutoAugmentPolicy.IMAGENET,
        interpolation: mindspore.dataset.vision.Inter = mindspore.dataset.vision.Inter.NEAREST,
        fill: Optional[List[float]] = None
    ) -> None:
        super().__init__()
        self.policy = policy
        self.interpolation = interpolation
        self.fill = fill
        self.policies = self._get_policies(policy)
        self.op_history = None  # 用于记录具体使用了哪种增强
        self.fixed = False

    def _get_policies(
        self,
        policy: AutoAugmentPolicy
    ) -> List[Tuple[Tuple[str, float, Optional[int]], Tuple[str, float, Optional[int]]]]:
        if policy == AutoAugmentPolicy.IMAGENET:
            return [
                (("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
                (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
                (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),
                (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
                (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),
                (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
                (("Rotate", 0.8, 8), ("Color", 0.4, 0)),
                (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
                (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
                (("Color", 0.8, 8), ("Solarize", 0.8, 7)),
                (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
                (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),
                (("Color", 0.4, 0), ("Equalize", 0.6, None)),
                (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),
                (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
                (("Invert", 0.6, None), ("Equalize", 1.0, None)),
                (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
                (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
            ]
        elif policy == AutoAugmentPolicy.CIFAR10:
            return [
                (("Invert", 0.1, None), ("Contrast", 0.2, 6)),
                (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
                (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
                (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
                (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
                (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
                (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
                (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
                (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
                (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
                (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
                (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
                (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
                (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
                (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
                (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
                (("Color", 0.9, 9), ("Equalize", 0.6, None)),
                (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
                (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
                (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
                (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
                (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
                (("Equalize", 0.8, None), ("Invert", 0.1, None)),
                (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
            ]
        elif policy == AutoAugmentPolicy.SVHN:
            return [
                (("ShearX", 0.9, 4), ("Invert", 0.2, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
                (("ShearY", 0.9, 8), ("Invert", 0.4, None)),
                (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
                (("Invert", 0.9, None), ("AutoContrast", 0.8, None)),
                (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
                (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),
                (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
                (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),
                (("Invert", 0.9, None), ("Equalize", 0.6, None)),
                (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),
                (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
                (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),
                (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
                (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),
                (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
                (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),
                (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
                (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),
                (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
                (("ShearX", 0.7, 2), ("Invert", 0.1, None)),
            ]
        else:
            raise ValueError("The provided policy {} is not recognized.".format(policy))

    def _augmentation_space(self, num_bins: int, image_size: List[int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "ShearX": (np.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (np.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (np.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "TranslateY": (np.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "Rotate": (np.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (np.linspace(0.0, 0.9, num_bins), True),
            "Color": (np.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (np.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (np.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (np.arange(num_bins) / ((num_bins - 1) / 4).round()), False),
            "Solarize": (np.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (0.0, False),
            "Equalize": (0.0, False),
            "Invert": (0.0, False),
        }

    @staticmethod
    def get_params(transform_num: int) -> Tuple[int, Tensor, Tensor]:
        """Get parameters for autoaugment transformation

        Returns:
            params required by the autoaugment transformation
        """
        policy_id = int(np.random.randint(0,transform_num, size=(1,)))

        probs = np.random.rand(2)

        signs = np.random.randint(0, 2, size=(2,))

        policy_id_tensor = Tensor(policy_id, dtype=mindspore.int32)
        probs_tensor = Tensor(probs, dtype=mindspore.float32)
        signs_tensor = Tensor(signs, dtype=mindspore.int32)

        return policy_id_tensor, probs_tensor, signs_tensor

    def construct(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: AutoAugmented image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * mindspore.dataset.vision.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        if self.fixed is False:
            transform_id, probs, signs = self.get_params(len(self.policies))
            for i, (op_name, p, magnitude_id) in enumerate(self.policies[transform_id]):
                if probs[i] <= p:
                    op_meta = self._augmentation_space(10, mindspore.dataset.vision.get_image_size(img))
                    magnitudes, signed = op_meta[op_name]

                    if magnitude_id is not None:
                        magnitude = float(magnitudes[magnitude_id])
                    else:
                        magnitude = 0.0
                    if signed and signs[i] == 0:
                        magnitude *= -1.0
                    img = apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
                    self.op_history = (op_name, magnitude)
            self.fixed = True  # 进行过一次增强后，固定增强参数
        else:
            if self.op_history is None:
                return img
            img = apply_op(img, self.op_history[0], self.op_history[1], interpolation=self.interpolation, fill=fill)

        return img

    def unfixed(self):
        self.op_history = None
        self.fixed = False

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(policy={}, fill={})'.format(self.policy, self.fill)


class RandAugment(nn.Cell):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        """

    def __init__(self, num_ops: int = 2, magnitude: int = 9, num_magnitude_bins: int = 31,
                 interpolation: mindspore.dataset.vision.Inter = mindspore.dataset.vision.Inter.NEAREST,
                 fill: Optional[List[float]] = None) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.op_history = []
        self.fixed = False

    def _augmentation_space(self, num_bins: int, image_size: List[int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (0.0, False),
            "ShearX": (np.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (np.linspace(0.0, 0.3, num_bins), True),
            "TranslateX": (np.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            "TranslateY": (np.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            "Rotate": (np.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (np.linspace(0.0, 0.9, num_bins), True),
            "Color": (np.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (np.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (np.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (np.arange(num_bins) / ((num_bins - 1) / 4)).round(), False),
            "Solarize": (np.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (0.0, False),
            "Equalize": (0.0, False),
        }
    
    def construct(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * mindspore.dataset.vision.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        if self.fixed is False:
            for _ in range(self.num_ops):
                op_meta = self._augmentation_space(self.num_magnitude_bins, mindspore.dataset.vision.get_image_size(img))
                op_index = int(np.random.randint(0,len(op_meta), size=(1,)))
                op_name = list(op_meta.keys())[op_index]
                magnitudes, signed = op_meta[op_name]
                if isinstance(magnitudes,float):
                    magnitude = 0.0
                else:
                    magnitude = float(magnitudes[self.magnitude]) 
                if signed and Tensor(np.random.randint(0, 2, (1,))):
                    magnitude *= -1.0
                img = apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
                self.op_history.append((op_name, magnitude))
            self.fixed = True
        else:
            for op_name, magnitude in self.op_history:
                img = apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img

    def unfixed(self):
        self.op_history = []
        self.fixed = False

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_ops={num_ops}'
        s += ', magnitude={magnitude}'
        s += ', num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)


class TrivialAugmentWide(nn.Cell):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        """

    def __init__(self,
                num_magnitude_bins: int = 31, 
                interpolation: mindspore.dataset.vision.Inter = mindspore.dataset.vision.Inter.NEAREST,
                fill: Optional[List[float]] = None) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.op_history = None
        self.fixed = False

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (0.0, False),
            "ShearX": (np.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (np.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (np.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (np.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (np.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (np.linspace(0.0, 0.99, num_bins), True),
            "Color": (np.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (np.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (np.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (8 - (np.arange(num_bins) / ((num_bins - 1) / 6)).round(), False),
            "Solarize": (np.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (0.0, False),
            "Equalize": (0.0, False),
        }

    def construct(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * mindspore.dataset.vision.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(np.random.randint(0, len(op_meta), size=(1,)))

        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        if isinstance(magnitudes,float):
            magnitude = 0.0
        else:
            magnitude = float(magnitudes[np.random.randint(0,len(magnitudes), (1,), dtype=np.int64)]) 
        if signed and Tensor(np.random.randint(0, 2, (1,))):
            magnitude *= -1.0

        if self.fixed is False:
            self.op_history = (op_name, magnitude)
            self.fixed = True
            return apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)
        else:
            return apply_op(img, self.op_history[0], self.op_history[1], interpolation=self.interpolation, fill=fill)

    def unfixed(self):
        self.op_history = None
        self.fixed = False

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)


if __name__ == '__main__':
    print('--- run autoaugment test ---')
    for i in range(10):
        x = mindspore.ops.randn([2, 3, 32, 32]).to(mindspore.uint8)
        augmentor = AutoAugment()
        x = augmentor(x)
        print(augmentor.op_history)
        x = augmentor(x)
        print(augmentor.op_history)
