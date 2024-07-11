# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import cv2
import argparse


import math
import time
import datetime
import matplotlib.pyplot as plt
import scipy.misc as misc


from natsort import natsort
#from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips


class Measure():
    def __init__(self, net='alex', use_gpu=False):
        self.device = 'cuda' if use_gpu else 'cpu'
       # self.model = lpips.LPIPS(net=net)
      #  self.model.to(self.device)

    def measure(self, imgA, imgB):
        return [float(f(imgA, imgB)) for f in [self.psnr, self.ssim ]]

    def lpips(self, imgA, imgB, model=None):
         ##input [0,255],BGR

        tA = t(imgA).to(self.device)
        tB = t(imgB).to(self.device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def psnr(self, sr, hr, scale=4, rgb_range=255.0, benchmark=True):
        ##input [0,255],BGR
        def _toTensor(x):
            x = x[:,:,[2,1,0]]
            x =torch.from_numpy(np.ascontiguousarray(x.transpose((2, 0, 1)))).clamp(0, 255).float()
            x = x.unsqueeze(0)
            return x

        sr = _toTensor(sr)
        hr = _toTensor(hr)
        diff = (sr - hr).data.div(rgb_range)
        if benchmark:
            shave = scale
            if diff.size(1) > 1:
                convert = diff.new(1, 3, 1, 1)
                convert[0, 0, 0, 0] = 65.738
                convert[0, 1, 0, 0] = 129.057
                convert[0, 2, 0, 0] = 25.064
                diff.mul_(convert).div_(256)
                diff = diff.sum(dim=1, keepdim=True)
        else:
            shave = scale + 6
        import math
        shave = math.ceil(shave)#+6
        valid = diff[:, :, shave:-shave, shave:-shave]
        mse = valid.pow(2).mean()
        if mse == 0:
            mse = mse + 1e-6
        return -10 * math.log10(mse)

    def ssim(self, img1, img2, scale=4, benchmark=True):

        def _toTensor(x):
            x = x[:,:,[2,1,0]]
            x =torch.from_numpy(np.ascontiguousarray(x.transpose((2, 0, 1)))).clamp(0, 255).float()
            x = x.unsqueeze(0)
            return x

        img1 = _toTensor(img1)
        img2 = _toTensor(img2)
        if benchmark:
            border = math.ceil(scale)# + 6
        else:
            border = math.ceil(scale) + 6

        img1 = img1.data.squeeze().float().clamp(0, 255).round().cpu().numpy()
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = img2.data.squeeze().cpu().numpy()
        img2 = np.transpose(img2, (1, 2, 0))

        img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 255.0 + 16.0
        img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 255.0 + 16.0
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        h, w = img1.shape[:2]
        img1_y = img1_y[border:h - border, border:w - border]
        img2_y = img2_y[border:h - border, border:w - border]

        if img1_y.ndim == 2:
            return ssim(img1_y, img2_y)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')



def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def format_result(psnr, ssim, lpips):
    return f'{psnr:0.2f}, {ssim:0.3f}, {lpips:0.3f}'

def measure_dirs(dirA, dirB, use_gpu, verbose=False):
    if verbose:
        vprint = lambda x: print(x)
    else:
        vprint = lambda x: None


    t_init = time.time()

    paths_A = fiFindByWildcard(os.path.join(dirA, f'*.{type}'))
    paths_B = fiFindByWildcard(os.path.join(dirB, f'*.{type}'))

    vprint("Comparing: ")
    vprint(dirA)
    vprint(dirB)

    measure = Measure(use_gpu=use_gpu)

    results = []
    for pathA, pathB in zip(paths_A, paths_B):
        result = OrderedDict()

        t = time.time()
        result['psnr'], result['ssim'], result['lpips'] = measure.measure(imread(pathA), imread(pathB))
        d = time.time() - t
        vprint(f"{pathA.split('/')[-1]}, {pathB.split('/')[-1]}, {format_result(**result)}, {d:0.1f}")

        results.append(result)

    psnr = np.mean([result['psnr'] for result in results])
    ssim = np.mean([result['ssim'] for result in results])
    lpips = np.mean([result['lpips'] for result in results])

    vprint(f"Final Result: {format_result(psnr, ssim, lpips)}, {time.time() - t_init:0.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dirA', default='.', type=str)
    parser.add_argument('-dirB', default='.', type=str)
    parser.add_argument('-type', default='png')
    parser.add_argument('--use_gpu', action='store_true', default=False)
    args = parser.parse_args()

    dirA = args.dirA
    dirB = args.dirB
    type = args.type
    use_gpu = args.use_gpu

    if len(dirA) > 0 and len(dirB) > 0:
        measure_dirs(dirA, dirB, use_gpu=use_gpu, verbose=True)
