import math
import PIL
from PIL import Image, ImageDraw


from typing import List, Optional

import numpy as np

import mindspore
from mindspore import Tensor
import mindspore.dataset.vision as vision


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=0.2):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if type(img) == Tensor:
            h = img.shape[1]
            w = img.shape[2]
            mask = np.ones((h, w), np.float32)
            for n in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                cut_size = int(min(w, h) * self.length)
                y1 = np.clip(y - cut_size // 2, 0, h)
                y2 = np.clip(y + cut_size // 2, 0, h)
                x1 = np.clip(x - cut_size // 2, 0, w)
                x2 = np.clip(x + cut_size // 2, 0, w)
                mask[y1: y2, x1: x2] = 0.
            mask = mindspore.Tensor.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask
        else:
            if isinstance(img, np.ndarray):
                img = vision.ToPIL()(img)
            w, h = img.size
            cut_size = int(min(w, h) * self.length)
            x0 = np.random.uniform(0, w)
            y0 = np.random.uniform(0, h)
            x0 = int(max(0, x0 - cut_size / 2.))
            y0 = int(max(0, y0 - cut_size / 2.))
            x1 = int(min(w, x0 + cut_size))
            y1 = int(min(h, y0 + cut_size))
            xy = (x0, y0, x1, y1)
            # gray
            color = (127, 127, 127)
            img = img.copy()
            PIL.ImageDraw.Draw(img).rectangle(xy, color)

        return img

def apply_op(img: Tensor, op_name: str, magnitude: float,
             interpolation: mindspore.dataset.vision.Inter, fill: Optional[List[float]]):
    
    if op_name == "ShearX":
        affine_transformer = vision.Affine(degrees=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0], 
                                           resample=interpolation, fill_value=fill)
        img = affine_transformer(img)

    elif op_name == "ShearY":
        affine_transformer = vision.Affine(degrees=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                                           resample=interpolation, fill_value=fill)
        img = affine_transformer(img)

    elif op_name == "TranslateX":

        if magnitude > 1 or magnitude <-1:
            magnitude /= 10
        affine_transformer = vision.Affine(degrees=0.0, translate=[int(magnitude), 0], scale=1.0, shear=[0.0, 0.0],
                                           resample=interpolation, fill_value=fill)
        img = affine_transformer(img)

    elif op_name == "TranslateY":
        if magnitude > 1 or magnitude <-1:
            magnitude /= 10
        affine_transformer = vision.Affine(degrees=0.0, translate=[0, int(magnitude)], scale=1.0, shear=[0.0, 0.0],
                                           resample=interpolation, fill_value=fill)
        img = affine_transformer(img)

    elif op_name == "Rotate":
        rotate_transformer = vision.Rotate(degrees=magnitude, resample=interpolation, fill_value=fill)
        img = rotate_transformer(img)

    elif op_name == "Brightness":
        adjust_brightness_transformer  = vision.AdjustBrightness(1.0 + magnitude)
        img = adjust_brightness_transformer(img)

    elif op_name == "Color":
        adjust_saturation_transformer  = vision.AdjustSaturation(1.0 + magnitude)
        img = adjust_saturation_transformer(img)

    elif op_name == "Contrast":
        adjust_contrast_transformer  = vision.AdjustContrast(1.0 + magnitude)
        img = adjust_contrast_transformer(img)

    elif op_name == "Sharpness":
        adjust_sharpness_transformer  = vision.AdjustSharpness(1.0 + magnitude)
        img = adjust_sharpness_transformer(img)

    elif op_name == "Posterize":
        posterize_transformer = vision.Posterize(int(magnitude))
        img = posterize_transformer(img)

    elif op_name == "Solarize":
        solarize_transformer = vision.Solarize(int(magnitude))
        img = solarize_transformer(img)

    elif op_name == "AutoContrast":
        autocontrast_transformer = vision.AutoContrast()
        img = autocontrast_transformer(img)

    elif op_name == "Equalize":
        autocontrast_transformer = vision.Equalize()
        img = autocontrast_transformer(img)

    elif op_name == "Invert":
        invert_transformer = vision.Invert()
        img = invert_transformer(img)
        
    elif op_name == "Identity":
        pass
    elif op_name == "CutOut":
        img = Cutout(length=magnitude)(img)
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img
