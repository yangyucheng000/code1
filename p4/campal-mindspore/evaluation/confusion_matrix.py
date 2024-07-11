import numpy as np
# import torch
import mindspore
from mindspore import Tensor


def calculate_confusion_matrix(pred, target):
    """Calculate confusion matrix according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).

    Returns:
        torch.Tensor: Confusion matrix with shape (C, C), where C is the number
             of classes.
    """

    if isinstance(pred, np.ndarray):
        # pred = torch.from_numpy(pred)
        pred = Tensor(pred)
    if isinstance(target, np.ndarray):
        # target = torch.from_numpy(target)
        target = Tensor(target)
    assert (
        isinstance(pred, Tensor) and isinstance(target, Tensor)), \
        (f'pred and target should be torch.Tensor or np.ndarray, '
         f'but got {type(pred)} and {type(target)}.')

    # num_classes = pred.size(1)
    num_classes = pred.shape(1)
    _, pred_label = pred.topk(1, dim=1)
    pred_label = pred_label.view(-1)
    target_label = target.view(-1)
    assert len(pred_label) == len(target_label)
    # confusion_matrix = torch.zeros(num_classes, num_classes)
    confusion_matrix = mindspore.ops.zeros(num_classes, num_classes)
    # with torch.no_grad():
    for t, p in zip(target_label, pred_label):
        confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix


def support(pred, target, average_mode='macro'):
    """Calculate the total number of occurrences of each label according to
        the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction with shape (N, C).
        target (torch.Tensor | np.array): The target of each prediction with
            shape (N, 1) or (N,).
        average_mode (str): The type of averaging performed on the result.
            Options are 'macro' and 'none'. If 'none', the scores for each
            class are returned. If 'macro', calculate metrics for each class,
            and find their unweighted sum.
            Defaults to 'macro'.

    Returns:
        float | np.array: Precision, recall, f1 score.
            The function returns a single float if the average_mode is set to
            macro, or a np.array with shape C if the average_mode is set to
             none.
    """
    confusion_matrix = calculate_confusion_matrix(pred, target)
    # with torch.no_grad():
    res = confusion_matrix.sum(1)
    if average_mode == 'macro':
        # res = float(res.sum().numpy())
        res = float(res.sum().asnumpy())
    elif average_mode == 'none':
        # res = res.numpy()
        res = res.asnumpy()
    else:
        raise ValueError(f'Unsupport type of averaging {average_mode}.')
    return res
