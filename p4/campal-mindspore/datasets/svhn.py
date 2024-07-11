# from torch.utils.data import Dataset
# from torchvision import datasets
# import torch
# from torchvision import transforms
from PIL import Image
import numpy as np
from .builder import DATASETS
from .base_dataset import BaseDataset
from query_strategies.utils import UnNormalize
import mindspore

@DATASETS.register_module()
class svhn(BaseDataset):
    def __init__(self,
                 data_path=None,
                 initial_size=None):
        super(svhn, self).__init__(data_path, initial_size)

    # Used for initialize data info
    def load_data(self):
        if self.DATA_PATH is None:
            self.DATA_PATH = '../data/SVHN'
        # raw_tr = datasets.SVHN(self.DATA_PATH, split='train', download=True)
        raw_tr = mindspore.dataset.SVHNDataset(self.DATA_PATH, usage='train')
        # raw_te = datasets.SVHN(self.DATA_PATH, split='test', download=True)
        raw_te = mindspore.dataset.SVHNDataset(self.DATA_PATH, usage='test')

        # num_tr = len(raw_tr.labels)
        num_tr = len(raw_tr)
        num_vl = 0
        # num_te = len(raw_te.labels)
        num_te = len(raw_te)
        class_to_idx_list = {i: [] for i in range(10)}
        val_idx_list = []


        # for idx, target in enumerate(raw_tr.labels):
        #     class_to_idx_list[int(target)].append(idx)

        raw_tr_data_iter = raw_tr.create_dict_iterator()

        for idx, data in enumerate(raw_tr_data_iter):
            target = data['label']
            class_to_idx_list[int(target)].append(idx)

        for _, class_elem in class_to_idx_list.items():
            val_idx_list.extend(class_elem[-1000:])
            num_vl += 1000
            num_tr -= 1000

        data_infos_train_full = []
        # 遍历数据集迭代器并逐个获取数据样本信息
        for i, data in enumerate(raw_tr_data_iter):
            if i >= (num_tr + num_vl):
                break  # 当达到训练集和验证集样本数量之和时结束循环
            image = data['image']  # 获取图像数据
            label = data['label']  # 获取标签数据
            sample_info = {'no': i, 'img': image, 'gt_label': label}  # 创建包含样本信息的字典
            data_infos_train_full.append(sample_info)  # 将样本信息字典添加到列表中

        self.DATA_INFOS['train_full'] = data_infos_train_full

        # self.DATA_INFOS['train_full'] = [{'no': i, 'img': raw_tr.data[i],
        #                                   'gt_label': raw_tr.labels[i].item()} for i in range(num_tr + num_vl)]
        self.DATA_INFOS['val'] = np.array(self.DATA_INFOS['train_full'])[val_idx_list].tolist()
        self.DATA_INFOS['train_full'] = np.delete(np.array(self.DATA_INFOS['train_full']), val_idx_list).tolist()
        # self.DATA_INFOS['val'] = [{'no': i, 'img': raw_te.data[i],
        #                            'gt_label': raw_te.labels[i].item()} for i in range(num_te)]
        # self.DATA_INFOS['test'] = [{'no': - (i + 1), 'img': raw_te.data[i],
        #                             'gt_label': raw_te.labels[i].item()} for i in range(num_te)]

        data_infos_test = []
        # 遍历数据集迭代器并逐个获取数据样本信息
        raw_te_data_iter = raw_te.create_dict_iterator()
        for i, data in enumerate(raw_te_data_iter):
            if i >= (num_te):
                break  # 当达到训练集和验证集样本数量之和时结束循环
            image = data['image']  # 获取图像数据
            label = data['label']  # 获取标签数据
            sample_info = {'no': - (i + 1), 'img': image, 'gt_label': label}  # 创建包含样本信息的字典
            data_infos_test.append(sample_info)  # 将样本信息字典添加到列表中
        self.DATA_INFOS['test'] = data_infos_test

        self.num_samples = num_tr + num_vl + num_te
        self.INDEX_LB = np.zeros(num_tr, dtype=bool)
        self.CLASSES = [str(i) for i in range(10)]

    # Only used for loading data
    def prepare_data(self, idx, split, transform=None, aug_transform=None):
        x, y = self.DATA_INFOS[split][idx]['img'], self.DATA_INFOS[split][idx]['gt_label']
        
        x_np = x.asnumpy()
        
        # print('xnp_shape',x_np.shape) (32,32,3)
        # print('xx',np.transpose(x_np, (1, 2, 0)).shape)32,32,3
        
        # x = Image.fromarray(np.transpose(x_np, (1, 2, 0)))
        x = Image.fromarray(x_np)
        # print('x',x.size, x.mode)(32,32,RGB)
        
        if aug_transform is not None:
            x = aug_transform.construct(x)
            # x = aug_transform(x)
        if transform is None:
            x = self.TRANSFORM[split](x)
        else:
            x = transform(x)
        return x, y, self.DATA_INFOS[split][idx]['no'], idx

    @property
    def default_train_transform(self):
        # return transforms.Compose([
        #     transforms.Pad(2, padding_mode='reflect'),
        #     transforms.RandomCrop(32),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
        #                          std=[0.1980, 0.2010, 0.1970])
        # ])
        return mindspore.dataset.transforms.Compose([
            mindspore.dataset.vision.Pad(2, padding_mode=mindspore.dataset.vision.Border.REFLECT),
            mindspore.dataset.vision.RandomCrop(32),
            mindspore.dataset.vision.RandomHorizontalFlip(),
            mindspore.dataset.vision.ToTensor(),
            mindspore.dataset.vision.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                 std=[0.1980, 0.2010, 0.1970],is_hwc = False)
        ])

    @property
    def default_val_transform(self):
        # return transforms.Compose([
        #     transforms.CenterCrop(32),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.4377, 0.4438, 0.4728],
        #                          std=[0.1980, 0.2010, 0.1970])
        # ])
        return mindspore.dataset.transforms.Compose([
            mindspore.dataset.vision.CenterCrop(32),
            mindspore.dataset.vision.ToTensor(),
            mindspore.dataset.vision.Normalize(mean=[0.4377, 0.4438, 0.4728],
                                 std=[0.1980, 0.2010, 0.1970],is_hwc = False)
        ])

    @property
    def inverse_transform(self):
        # return transforms.Compose([
        #     UnNormalize(mean=[0.4377, 0.4438, 0.4728],
        #                 std=[0.1980, 0.2010, 0.1970]),
        #     transforms.ToPILImage()
        # ])
        return mindspore.dataset.transforms.Compose([
            UnNormalize(mean=[0.4377, 0.4438, 0.4728],
                        std=[0.1980, 0.2010, 0.1970],is_hwc = False),
            mindspore.dataset.vision.ToPIL()
        ])

    def get_raw_data(self, idx, split='train'):
        transform = self.default_val_transform
        x = self.DATA_INFOS[split][idx]['img']
        x = Image.fromarray(np.transpose(x, (1, 2, 0)))
        x = transform(x)
        return x
