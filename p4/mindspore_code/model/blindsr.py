
import model.common as common
from mindspore.ops import functional as F
from mindspore import Tensor, context
import mindspore.ops as ops
import mindspore.ops as P

import mindspore.nn as nn

from mindspore.ops import functional as F

def make_model(args):
    return BlindSR(args)


class DA_conv(nn.Cell):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.kernel = nn.SequentialCell(
            nn.Dense(64, 64, has_bias=False),
            nn.LeakyReLU(0.1),
            nn.Dense(64, 64 * self.kernel_size * self.kernel_size, has_bias=False),
            nn.Sigmoid()
        )
        self.conv = common.default_conv(channels_in, channels_out, 1)
        self.ca = CA_layer(channels_in, channels_out, reduction)

        self.relu = nn.LeakyReLU(0.1)

    def construct(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].shape

        # branch 1
        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size, self.kernel_size)
        out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, pad_mode='pad',padding=(self.kernel_size-1)//2))
        out = self.conv(out.view(b, -1, h, w))

        # branch 2
        out = out + self.ca(x)

        return out


class CA_layer(nn.Cell):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.SequentialCell(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, pad_mode='pad', padding=0, has_bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, pad_mode='pad',padding=0, has_bias=False),
            nn.Sigmoid()
        )

    def construct(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(x[1][:, :, None, None])

        return x[0] * att

class SA_layer(nn.Cell):
    def __init__(self):
        super(SA_layer, self).__init__()
        self.conv_du = nn.SequentialCell(
            nn.Conv2d(2, 1, 1, 1,  pad_mode='pad',padding=0, has_bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1 , 1, 1, 1,  pad_mode='pad',padding=0, has_bias=False),
            nn.Sigmoid()
        )
        self.cat = ops.Concat(1)
        self.mean = ops.ReduceMean(keep_dims=True)
        self.max = ops.ArgMaxWithValue(1,keep_dims=True)
    def construct(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        avg_out  =  self.mean(x,1)
        _,max_out = self.max(x)
        at = self.cat((avg_out,max_out))

        att = self.conv_du(at)
        return x * att
class SA_CA(nn.Cell):
    def __init__(self, conv, n_feat, reduction):
        super(SA_CA, self).__init__()
        self.ca = CA_layer(n_feat, n_feat, reduction)
        self.sa = SA_layer()

    def construct(self, x):

        input = x[0]
        deg = x[1]
        out = self.sa(input)
        out = self.ca([out, deg])

        return out
class DAB(nn.Cell):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DAB, self).__init__()

        self.da_conv1 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.da_conv2 = DA_conv(n_feat, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)
        self.ca_sa = SA_CA(n_feat,n_feat,reduction )

        self.relu =  nn.LeakyReLU(0.1)

    def construct(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        input = x[0]
        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2([out, x[1]]))
        input = self.ca_sa([input, x[1]])
        out = self.conv2(out) + input#x[0]

        return out


class DAG(nn.Cell):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_blocks):
        super(DAG, self).__init__()
        self.n_blocks = n_blocks
        modules_body = [
            DAB(conv, n_feat, kernel_size, reduction) \
            for _ in range(n_blocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.ca_sa = SA_CA(n_feat,n_feat,reduction )

        self.body = nn.SequentialCell(*modules_body)

    def construct(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        res = x[0]
        input = x[0]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1]])
        res = self.body[-1](res)
        input = self.ca_sa([input, x[1]])
        res = res + input

        return res
class DASR(nn.Cell):
    def __init__(self, args, conv=common.default_conv):
        super(DASR, self).__init__()

        self.n_groups = 5
        n_blocks = 5
        n_feats = 64
        kernel_size = 3
        reduction = 8
        scale = int(args.scale[0])

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        rgb_range = 255
        self.sub_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(rgb_range, rgb_mean, rgb_std, 1)
        # head module
        modules_head = [conv(3, n_feats, kernel_size)]
        self.head = nn.SequentialCell(*modules_head)

        # compress
        self.compress = nn.SequentialCell(
            nn.Dense(256, 64, has_bias=False),
            nn.LeakyReLU(0.1)
        )

        # body
        modules_body = [
            DAG(common.default_conv, n_feats, kernel_size, reduction, n_blocks) \
            for _ in range(self.n_groups)
        ]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        self.body = nn.SequentialCell(*modules_body)
        self.sa_ca = SA_CA(n_feats,n_feats,reduction )

        # tail
        modules_tail = [
                        conv(n_feats, 3, kernel_size)]
        self.tail = nn.SequentialCell(*modules_tail)
        self.up_conv1 = conv(n_feats, 4 * n_feats, 3)
        self.up_conv2 = conv(n_feats, 4 * n_feats, 3)

        self.up =  ops.DepthToSpace(2)
        
    def construct(self, x, k_v):
        k_v = self.compress(k_v)
        print_op = P.Print()

     #   print_op(x)

        # sub mean
        x = self.sub_mean(x)

        # head
        x = self.head(x)

        # body
        res = x
        for i in range(self.n_groups):
            res = self.body[i]([res, k_v])
        res = self.body[-1](res)
        x = self.sa_ca([x, k_v])

        res = res + x
        up_1 = self.up_conv1(res)
        up_1 =  self.up(up_1)
        up_2 = self.up_conv2(up_1)
        up_2 =  self.up(up_2)

        x = self.tail(up_2)

        x = self.add_mean(x)

        return x
class Encoder(nn.Cell):
    def __init__(self):
        super(Encoder, self).__init__()


        self.E_simple = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=3, pad_mode='pad',padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3,pad_mode='pad', padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, pad_mode='pad',padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, pad_mode='pad',padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)    )
        self.E = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=3,pad_mode='pad', padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, pad_mode='pad',padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, pad_mode='pad',padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, pad_mode='pad',padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1)

        )
        self.mlp_fouse = nn.SequentialCell(
            nn.Dense(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dense(256, 256),
        )
        self.ap = ops.AdaptiveAvgPool2D((1 ,1))
        self.mlp = nn.SequentialCell(
            nn.Dense(256, 256),
            nn.LeakyReLU(0.1),
            nn.Dense(256, 256),
        )
        self.flatten = nn.Flatten()
        self.cat = ops.Concat(1)

    def construct(self, x ,is_image=False):
        img = x[0]
        mask = x[1]
        if not is_image:
            b,c,h,w =  mask.shape
            new_h,new_w = int(h //4), int(w // 4)
            mask =  ops.interpolate(mask,size=(new_h,new_w))# scale_factor=(0.25,0.25))
        complex_part = img * mask
        simple_part = img * (1 - mask)
        simple_fea =  self.flatten(self.ap( self.E_simple(simple_part) ))
        complex_fea =  self.flatten(self.ap( self.E(complex_part)))
        lin = self.cat ((simple_fea , complex_fea))
        fea = self.mlp_fouse(self.cat ((simple_fea , complex_fea)))

        out = self.mlp(fea)
        return out

class BlindSR(nn.Cell):
    def __init__(self, args):
        super(BlindSR, self).__init__()

        # Generator
        self.G = DASR(args)

        # Encoder
        self.E = Encoder()

    def construct(self, x , mask,is_image=False):

        fea = self.E([ x , mask] , is_image)

        sr = self.G(x, fea)

        return sr