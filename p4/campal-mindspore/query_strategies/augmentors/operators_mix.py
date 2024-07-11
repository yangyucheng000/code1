
import numpy as np
import mindspore
from mindspore import Tensor


def get_one_hot_label(labels=None, num_classes=10):
    # labels输出是一个Tensor
    if labels.shape == ():
        labels = labels.asnumpy().item()
        numpy_result = np.zeros(num_classes)
        numpy_result[labels] = 1
        return numpy_result


    labels_np = labels.asnumpy()
    numpy_result = np.zeros((labels_np.shape[0], num_classes))
    numpy_result[np.arange(labels_np.shape[0]), labels_np] = 1
    return numpy_result




def onehot(size, target):
    vec = np.zeros(size)
    vec[target] = 1.
    vec = mindspore.Tensor(vec, mindspore.float32)
    return vec


def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def MixUp(img_1, img_2, label_1, label_2, num_class, magnitude=None, alpha=1.0, is_bias=False):
    if magnitude is None:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        if is_bias:
            lam = max(lam, 1 - lam)
    else:
        lam = magnitude
    label_1, label_2 = get_one_hot_label(label_1, num_class),get_one_hot_label(label_2, num_class)
    img = lam * img_1 + (1 - lam) * img_2
    label = lam * label_1 + (1 - lam) * label_2
    return img, label


def CutMix(img_1, img_2, label_1, label_2, num_class, magnitude=None, alpha=1.0, is_bias=False):
    if magnitude is None:
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        if is_bias:
            lam = max(lam, 1 - lam)
    else:
        lam = magnitude
    # img_1 是numpy.ndarray
    bbx1, bby1, bbx2, bby2 = rand_bbox(img_1.shape, lam)
    img_1[:, bbx1:bbx2, bby1:bby2] = img_2[:, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_1.shape[-1] * img_1.shape[-2]))
    label_1, label_2 = get_one_hot_label(label_1, num_class), get_one_hot_label(label_2, num_class)
    label = label_1 * lam + label_2 * (1. - lam)
    return img_1, label


def apply_op(img_1, img_2, label_1, label_2, num_class, op_name, magnitude):
    if op_name == "MixUp":
        img, label = MixUp(img_1, img_2, label_1, label_2, num_class, magnitude)
    elif op_name == "CutMix":
        
        img, label = CutMix(img_1, img_2, label_1, label_2, num_class, magnitude)
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img, label
