# import torch.nn as nn
# import torch.nn.init as init
# from torchvision.models.resnet import resnet18
import mindspore
from mindspore import nn, Tensor
from mindspore.common.initializer import initializer, Normal, XavierNormal, Constant, Orthogonal
from mindspore import Parameter

def weight_init(m):
    """This is for initialing model parameters.

    Modules of type `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`,
    `nn.ConvTranspose1d`, `nn.ConvTranspose2d`, `nn.ConvTranspose3d`,
    `nn.BatchNorm1d`, `nn.BatchNorm2d`, `nn.BatchNorm3d`,
    `nn.Linear`, `nn.LSTM`, `nn.LSTMCell`, `nn.GRU`, `nn.GRUCell`
    will be initialized.

    Args:
        m (:obj:`torch.nn.Module`):
            The model to use

    Example:
        >>> model = resnet18()
        >>> model.apply(weight_init)

    """
    if isinstance(m, nn.Conv1d):
        # init.normal_(m.weight.data) # mean = 0 std = 1
        m.weight.data = Parameter(initializer(Normal()))
        if m.bias is not None:
            # init.normal_(m.bias.data)
            m.bias.data = Parameter(initializer(Normal()))
    elif isinstance(m, nn.Conv2d):
        # init.xavier_normal_(m.weight.data)
        m.weight.data = Parameter(initializer(XavierNormal()))
        if m.bias is not None:
            # init.normal_(m.bias.data)
            m.bias.data = Parameter(initializer(Normal()))
    elif isinstance(m, nn.Conv3d):
        # init.xavier_normal_(m.weight.data)
        m.weight.data = Parameter(initializer(XavierNormal()))
        if m.bias is not None:
            # init.normal_(m.bias.data)
            m.bias.data = Parameter(initializer(Normal()))

    # elif isinstance(m, nn.ConvTranspose1d):
    elif isinstance(m, nn.Conv1dTranspose):
        # init.normal_(m.weight.data)
        m.weight.data = Parameter(initializer(Normal()))
        if m.bias is not None:
            # init.normal_(m.bias.data)
            m.bias.data = Parameter(initializer(Normal()))

    # elif isinstance(m, nn.ConvTranspose2d):
    elif isinstance(m, nn.Conv2dTranspose):
        # init.xavier_normal_(m.weight.data)
        m.weight.data = Parameter(initializer(XavierNormal()))
        if m.bias is not None:
            # init.normal_(m.bias.data)
            m.bias.data = Parameter(initializer(Normal()))

    # elif isinstance(m, nn.ConvTranspose3d):
    elif isinstance(m, nn.Conv3dTranspose):
        # init.xavier_normal_(m.weight.data)
        m.weight.data = Parameter(initializer(XavierNormal()))
        if m.bias is not None:
            # init.normal_(m.bias.data)
            m.bias.data = Parameter(initializer(Normal()))

    elif isinstance(m, nn.BatchNorm1d):
        # init.normal_(m.weight.data, mean=1, std=0.02)
        m.weight.data = Parameter(initializer(Normal(sigma=0.02, mean=1.0)))
        # init.constant_(m.bias.data, 0)
        m.bias.data = Parameter(initializer(Constant(0)))
    elif isinstance(m, nn.BatchNorm2d):
        # init.normal_(m.weight.data, mean=1, std=0.02)
        m.weight.data = Parameter(initializer(Normal(sigma=0.02, mean=1.0)))
        # init.constant_(m.bias.data, 0)
        m.bias.data = Parameter(initializer(Constant(0)))
    elif isinstance(m, nn.BatchNorm3d):
        # init.normal_(m.weight.data, mean=1, std=0.02)
        m.weight.data = Parameter(initializer(Normal(sigma=0.02, mean=1.0)))
        # init.constant_(m.bias.data, 0)
        m.bias.data = Parameter(initializer(Constant(0)))
    # elif isinstance(m, nn.Linear):
    elif isinstance(m, nn.Dense):
        # init.xavier_normal_(m.weight.data)
        m.weight.data = Parameter(initializer(XavierNormal()))
        # init.normal_(m.bias.data)
        m.bias.data = Parameter(initializer(Normal()))
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                # init.orthogonal_(param.data)
                param.data = Parameter(initializer(Orthogonal()))
            else:
                # init.normal_(param.data)
                param.data = Parameter(initializer(Normal()))

    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                # init.orthogonal_(param.data)
                param.data = Parameter(initializer(Orthogonal()))
            else:
                # init.normal_(param.data)
                param.data = Parameter(initializer(Normal()))

    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                # init.orthogonal_(param.data)
                param.data = Parameter(initializer(Orthogonal()))
            else:
                # init.normal_(param.data)
                param.data = Parameter(initializer(Normal()))

    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                # init.orthogonal_(param.data)
                param.data = Parameter(initializer(Orthogonal()))

            else:
                # init.normal_(param.data)
                param.data = Parameter(initializer(Normal()))



if __name__ == '__main__':
    pass
