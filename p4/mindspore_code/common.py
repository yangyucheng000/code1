import random
import numpy as np
import skimage.color as sc
import torch
import numpy as np
import cv2 as cv
def get_mask(img,sub_size=21):
    img = img / 255.0
    all_mea , all_var =cv.meanStdDev(img)
    all_var_max = all_var.min()
    theat = 0.08

    scale = 1
    h,w,c= img.shape
    all_sub_image = []
    crop_sz = sub_size
    step = sub_size
    thres_sz = 0
    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)
    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            crop_img = img[x:x + crop_sz, y:y + crop_sz , : ]
            mea , var =cv.meanStdDev(crop_img)
            var_max = var.min()

            save_image  =  None
            if var_max > theat:
                save_image = np.ones((crop_sz,crop_sz,3))
            else:
                save_image = np.zeros((crop_sz,crop_sz,3))
            subimage_data = {
                "image":save_image,
                "x1":x*scale,
                "y1":(x+crop_sz)*scale,
                "x2":y*scale,
                "y2":(y+crop_sz)*scale
            }
            all_sub_image.append(subimage_data)
    new_image = make_up_image(all_sub_image,h,w,sub_size,scale)
    return new_image
def make_up_image(input_data,height,weight,sub_size,scale):
    img = np.zeros((height*scale,weight*scale , 3))
    img_quan = np.zeros((height*scale,weight*scale ,3 ))
    quan_one = np.ones((sub_size*scale,sub_size*scale, 3))
    for data in input_data:
        tmp_image = data['image']
        img[data['x1']:data['y1'],data['x2']:data['y2']] = img[data['x1']:data['y1'],data['x2']:data['y2'] , :] + tmp_image
        img_quan[ data['x1']:data['y1'],data['x2']:data['y2']] = img_quan[data['x1']:data['y1'],data['x2']:data['y2'] , :] + quan_one
    img = (img) / (img_quan)

    return img

def get_patch(img, patch_size=48, scale=1):
    th, tw = img.shape[:2] 
    tp = round(scale * patch_size)

    tx = random.randrange(0, (tw-tp))
    ty = random.randrange(0, (th-tp))

    return img[ty:ty + tp, tx:tx + tp, :]



def set_channel(img, n_channels=3):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    c = img.shape[2]
    if n_channels == 1 and c == 3:
        img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
    elif n_channels == 3 and c == 1:
        img = np.concatenate([img] * n_channels, 2)

    return img
def np2Tensor_mask(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))

    return np_transpose


def np2Tensor(img, rgb_range=255):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))


    return np_transpose


def augment(img, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    if hflip: img = img[:, ::-1, :]
    if vflip: img = img[::-1, :, :]
    if rot90: img = img.transpose(1, 0, 2)

    return img

