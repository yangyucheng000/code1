
from PIL import Image
import torch
import os
import cv2 as cv
import numpy as np
import scipy.misc
import torch
import pdb
import random
import common
import pickle
import numpy as np
import imageio
import mindspore.dataset as ds

class TestDataset():
    def __init__(self,args):
        image_path =  args.test_dir +  '/' + args.data_test+'/HR'
        self.files = os.listdir(image_path)
        self.hr_file_path = image_path
        self.args = args
    def __getitem__(self,index):
        hr, filename = self._load_file(index)
        hr = self.get_patch(hr)
        masks = [common.get_mask(img , sub_size=self.args.mask_patch) for img in hr]
        hr = [common.set_channel(img, n_channels=self.args.n_colors) for img in hr]
        hr_tensor = [common.np2Tensor(img, rgb_range=self.args.rgb_range)
                     for img in hr]
        masks_tensor = [common.np2Tensor_mask(j, rgb_range=self.args.rgb_range)
                     for j in masks]
        
        return np.stack(hr_tensor, 0), np.stack(masks_tensor, 0) ,filename


    def __len__(self):
        return len(self.files)

    def _load_file(self, idx):
        f_hr = self.hr_file_path + '/' + self.files[idx]
        filename =  self.files[idx]
        hr = imageio.imread(f_hr)
        return hr, filename

    def get_patch(self, hr):
        scale = self.args.scale
        out = [hr]
        return out
def get_loader(args):
    test_set = TestDataset(args)
    eval_ds = ds.GeneratorDataset(test_set,  ["HR", "mask",  "name"],shuffle=False)
    eval_ds = eval_ds.batch(1, drop_remainder=True)
    return None, eval_ds
