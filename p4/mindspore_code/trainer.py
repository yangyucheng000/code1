import os
import utility
import torch
from decimal import Decimal
from utils import util
from Measure import Measure
import cv2 as cv
import numpy as np
import numpy as np
import math
from mindspore.common import dtype as mstype

import random
import common
import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint
import mindspore.ops as ops
from mindspore import Tensor, context


def write_log_flie(txt_name,add_log):
        with open('log/' + txt_name,'a') as f:
                f.write(add_log)



def tensor2img(tensor,corp_size=0 ,out_type=np.uint8, min_max=(0, 255)):
    img_np = np.transpose(tensor[[2, 1, 0], :, :], (1, 2, 0))  
    return img_np.astype(out_type)
class Trainer():
    def __init__(self, args, loader, my_model, ckp):
        self.args = args
        self.scale = args.scale
        self.mea = Measure()

        self.ckp = ckp
        self.loader_train , self.loader_test  = loader

        self.model = my_model
        if self.args.pre_train != '.':
            param_dict = load_checkpoint(args.pre_train)
            load_param_into_net(self.model, param_dict)
            print("Load net weight successfully")


    def test(self):
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.set_train(False)

        timer_test = utility.timer()

        with torch.no_grad():
            if True:
                eval_psnr = 0
                eval_ssim = 0
                degrade = util.SRMDPreprocessing(
                    self.scale[0],
                    kernel_size=self.args.blur_kernel,
                    blur_type=self.args.blur_type,
                    sig=self.args.sig,
                    lambda_1=self.args.lambda_1,
                    lambda_2=self.args.lambda_2,
                    theta=self.args.theta,
                    noise=self.args.test_noise
                )

                for idx_img, (hr,  masks , filename) in enumerate(self.loader_test):
                    hr = hr.asnumpy()
                    hr =  torch.from_numpy(hr).float()
                    hr = self.crop_border(hr, self.scale[0])
                    lr, b_kernels = degrade(hr, random=False)  
                    lr = Tensor(lr[:,0,...].numpy() , mstype.float32)
                    hr = Tensor(hr[:,0,...].numpy(), mstype.float32)
                    masks = Tensor(masks[:,0,...].numpy(), mstype.float32)

                    sr = self.model(lr  , masks,False)

                    pred_np = np.squeeze(sr.asnumpy())
                    pred_np = utility.quantize(pred_np, 255)
                    sr = tensor2img(pred_np)

                    hr = np.squeeze(hr.asnumpy())
                    hr = utility.quantize(hr, 255)
                    hr = tensor2img(hr) 

                    psnr,ssim = self.mea.measure(hr,sr)                                       

                    timer_test.tic()
                    timer_test.hold()

                    eval_psnr += psnr
                    eval_ssim += ssim

                ave_psnr = round(eval_psnr / len(self.loader_test) , 2)


                ave_ssim = round(eval_ssim / len(self.loader_test) , 4)


                save_info =  '  testet: '+ self.args.data_test + ' epoch ' + str(epoch) + '  sig: ' + str(self.args.sig) + ' Noise :' + str(self.args.test_noise) +' psnr: '+str( round( eval_psnr  / len(self.loader_test) , 2) )   +  ' ssim: '+str( round( eval_ssim  / len(self.loader_test) , 4))   +'\n' #+' ssim: '+str(ave_ssim)   + '_lpips_'+str(ave_lpips) + '\n'
                print(save_info)
                write_log_flie(self.args.log_name,save_info)


    def crop_border(self, img_hr, scale):
        b, n, c, h, w = img_hr.size()

        img_hr = img_hr[:, :, :, :int(h//scale*scale), :int(w//scale*scale)]

        return img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs_encoder + self.args.epochs_sr

