
import mindspore
from mindspore import Tensor, nn
from typing import Tuple, Dict, List
from .operators_mix import apply_op
import numpy as np


class StrengthGuidedAugmentMixing():
    r"""Strength guided data augmentation for image mixing augmentations
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        """

    def __init__(self, magnitude: int = 9, num_magnitude_bins: int = 26, ablation_aug=None) -> None:
        super().__init__()
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.op_history = []
        self.fixed = False
        self.ablation_aug = ablation_aug

    def _augmentation_space(self, num_bins: int, image_size: List[int]) -> Dict[str, Tuple[Tensor, bool]]:
        # add image_size: List[int] if necessary
        return {
            # op_name: (magnitudes, signed)
            "MixUp": (np.linspace(0.0, 0.5, num_bins), False),
            "CutMix": (np.linspace(0.0, 0.5, num_bins), False)
        }

    def construct(self, img_1: Tensor, img_2: Tensor, label_1: Tensor, label_2: Tensor, num_class)\
            -> Tuple[Tensor, Tensor]:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        # 这里的img_1,img_2类型是tuple,里面的元素是np.ndarray
        img_1 = img_1[0]
        img_2 = img_2[0]

        if self.fixed is False:
            op_meta = self._augmentation_space(self.num_magnitude_bins, mindspore.dataset.vision.get_image_size(img_1))
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
            img = apply_op(img_1, img_2, label_1, label_2, num_class, op_name, magnitude)
            self.op_history.append((op_name, magnitude))
            self.fixed = True
        else:
            for op_name, magnitude in self.op_history:
                img = apply_op(img_1, img_2, label_1, label_2, num_class, op_name, magnitude)

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
