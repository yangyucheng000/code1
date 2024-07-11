import warnings
import numpy as np
import mindspore



def get_lr(optimizer,epoch):
    return  optimizer.get_lr().asnumpy()



def get_initialized_module(net, lr, momentum, weight_decay, milestones, optim_type='sgd', **kwargs):
    clf = net(**kwargs)
    learning_rates = [0.1, 0.01, 0.001] # gamma = 0.1 lr_initial=0.1
    lrs = mindspore.nn.dynamic_lr.piecewise_constant_lr(milestone=milestones,learning_rates=learning_rates)

    if optim_type == 'sgd':
        optimizer = mindspore.nn.SGD(clf.trainable_params(),  learning_rate=lr, momentum=momentum,
                              weight_decay=weight_decay)
    elif optim_type == 'adam':
        optimizer = mindspore.nn.Adam(clf.trainable_params(), learning_rate=lr)
    else:
        optimizer = mindspore.nn.AdamWeightDecay(clf.trainable_params(), learning_rate=lr)
        
    scheduler = mindspore.nn.piecewise_constant_lr(milestone=milestones,learning_rates=learning_rates)
    return clf, optimizer, scheduler


def get_images(c, n, indices_class, dataset):  # get random n images from class c
    indices = indices_class[c]
    if 0 < len(indices_class[c]) < n:
        indices = np.repeat(indices, n // len(indices_class[c]) + 1)
    elif len(indices_class[c]) == 0:
        warnings.warn(f"No samples in class {dataset.CLASSES[c]}!")
        return mindspore.ops.zeros([0, *tuple(dataset.get_raw_data(0).shape)])

    idx_shuffle = np.random.permutation(indices)[:n]
    data = mindspore.ops.stack([dataset.get_raw_data(idx, 'train') for idx in idx_shuffle])
    return data


class UnNormalize(mindspore.nn.Cell):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be denormalized.
        Returns:
            Tensor: Denormalized image.
        """
        t = tensor.clone()
        inv_mean = [-m/s for m, s in zip(self.mean, self.std)]
        inv_std = [1/s for s in self.std]
        normalize = mindspore.dataset.vision.Normalize(mean=inv_mean, std=inv_std,is_hwc=False)
        return normalize(t)


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return mindspore.Tensor(0, dtype=mindspore.float32)

    dis_weight = mindspore.ops.sum(1 - mindspore.ops.sum(gwr * gws, dim=-1) / (mindspore.ops.norm(gwr, dim=-1) * mindspore.ops.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def match_loss(gw_syn, gw_real, dis_metric):
    dis = mindspore.Tensor(0.0)

    if dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = mindspore.ops.cat(gw_real_vec, dim=0)
        gw_syn_vec = mindspore.ops.cat(gw_syn_vec, dim=0)
        dis = mindspore.ops.sum((gw_syn_vec - gw_real_vec)**2)

    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = mindspore.ops.cat(gw_real_vec, dim=0)
        gw_syn_vec = mindspore.ops.cat(gw_syn_vec, dim=0)
        dis = 1 - mindspore.ops.sum(gw_real_vec * gw_syn_vec, dim=-1) / (mindspore.ops.norm(gw_real_vec, dim=-1) * mindspore.ops.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s' % dis_metric)

    return dis


def get_one_hot_label(labels=None, num_classes=10):
    # labels输出是一个Tensor

    if labels.shape == ():
        labels = labels.asnumpy().item()
        numpy_result = np.zeros(num_classes)
        numpy_result[labels] = 1
        result = mindspore.Tensor(numpy_result, mindspore.float32)
        return result
    labels_np = labels.asnumpy()
    numpy_result = np.zeros((labels_np.shape[0], num_classes))
    numpy_result[np.arange(labels_np.shape[0]), labels_np] = 1
    result = mindspore.Tensor(numpy_result,mindspore.float32)
    return result





def soft_cross_entropy(pred, soft_targets):
    """A method for calculating cross entropy with soft targets"""
    logsoftmax = mindspore.nn.LogSoftmax()
    return mindspore.ops.mean(mindspore.ops.sum(- soft_targets * logsoftmax(pred), 1))


def print_class_object(obj, name, logger):
    for key, value in obj.__dict__.items():
        logger.info("CONFIG -- {} - {}: {}".format(name, key, value))
