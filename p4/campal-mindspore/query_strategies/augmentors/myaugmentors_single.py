from typing import List, Tuple, Optional, Dict
from .operators_single import apply_op

import mindspore
from mindspore import nn, Tensor
import numpy as np


class StrengthGuidedAugmentSingle():
    r"""Strength guided data augmentation for single-image augmentations modified from
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
                 fill: Optional[List[float]] = 0, ablation_aug=None) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.op_history = []
        self.fixed = False
        self.ablation_aug = ablation_aug

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
            "CutOut": (np.linspace(0.0, 0.5, num_bins), False)
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
                if self.ablation_aug is None:
                    op_index = int(np.random.randint(0, len(op_meta), size=(1,)))
                    op_name = list(op_meta.keys())[op_index]
                else:
                    op_name = self.ablation_aug
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
