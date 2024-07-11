import numpy as np
from pathlib import Path
import os
from getpass import getuser
from socket import gethostname
from utils.progressbar import track_iter_progress
from utils.text import TextLogger
from utils.timer import Timer
from .builder import STRATEGIES
from datasets.dataloader import GetHandler, Handler
from copy import deepcopy
from datasets.base_dataset import BaseDataset
from evaluation import *
from .utils import get_initialized_module, get_lr
import csv
import time
import pdb
import itertools
from scipy.special import softmax
import pickle
import mindspore
from mindspore import Tensor,nn


@STRATEGIES.register_module()
class Strategy:
    def __init__(self, dataset: BaseDataset, net, args, logger, timestamp):
        self.dataset = dataset
        self.net = net  # This is a function that returns a model structure, not the model itself!
        self.args = args
        # Model
        self.clf, self.optimizer, self.scheduler = None, None, None
        self.init_clf()
        # cycle info
        self.cycle_info = None
        self.init_cycle_info()
        # This is for resume
        self.cycle = 0
        self.epoch = 0
        # 两个logger比较复杂，之后再说
        # 这主要用来监控性能变化
        self.logger = logger
        self.TextLogger = TextLogger(self.clf, vars(args), logger)
        # 用来统计训练所消耗的时间
        self.timer = Timer()
        # 记录训练开始的时间戳
        self.timestamp = timestamp
        # 记录验证集/测试集每一轮的性能
        self.acc_val_list = []
        self.acc_test_list = []
        # 记录每一轮标注池样本数量
        self.num_labels_list = []
        # 输出主动学习初始配置
        self.TextLogger._dump_log(vars(args))

    # 用于在主动学习的每一轮初始化模型(retraining)，不用这个就是updating
    def init_clf(self):
        """When we want to initialize the model we use, apply this function.
        Random parameter initialization is included.
        """
        self.clf, self.optimizer, self.scheduler = \
            get_initialized_module(self.net, self.args.lr, self.args.momentum, self.args.weight_decay,
                                   self.args.milestones, num_classes=len(self.dataset.CLASSES))

    def init_cycle_info(self):
        # can include multiple aug infos, added at the end of the dict
        # score只对无标注样本有意义
        self.cycle_info = [{'no': i,
                            'class': self.dataset.CLASSES[int(self.dataset.DATA_INFOS['train_full'][i]['gt_label'])],
                            'label': int(self.dataset.INDEX_LB[i]),
                            'queried': 0,
                            'score': 0}
                           for i in range(len(self.dataset.DATA_INFOS['train_full']))]

    # 主动学习的核心函数，根据策略的不同而不同
    def query(self, n):
        """Query new samples according the current model or some strategies.

        :param n: (int)The number of samples to query.

        Returns:
            list[int]: The indices of queried samples.

        """
        raise NotImplementedError

    # 更新样本池
    def update(self, n, aug_args_list=None):
        if n == 0:
            return None

        # 接下来的任务是使用这个
        if aug_args_list is None:
            if self.args.aug_ulb_evaluation_mode in ['StrengthGuidedAugment', 'RandAugment']:
                aug_args_list = self.augment_optimizer_unlab()

        idxs_q = self.query(n, aug_args_list)
        # 目前对应基于无标注子集的顺序，所以现在必须修正至真正的下标

        
        idxs_q_np = np.array(idxs_q)
        idxs_q_np = np.array([tensor.item() for tensor in idxs_q_np])
        idxs_q = np.arange(len(self.dataset.DATA_INFOS['train_full']))[self.dataset.INDEX_ULB][idxs_q_np]
        self.dataset.update_lb(idxs_q)
        
        return idxs_q

    
    def _train(self, loader_tr, clf_group: dict, clf_name='train', soft_target=False, log_show=True):
        """Represents one epoch.

        :param loader_tr: (:obj:`torch.utils.data.DataLoader`) The training data wrapped in DataLoader.

        Accuracy and loss in the each iter will be recorded.

        """
        iter_out = self.args.out_iter_freq
        loss_list = []
        right_count_list = []
        samples_per_batch = []
        self.clf.set_train(True)

        optimizer = clf_group['optimizer']
        
                
        def train_step(data, label):
            (loss, out), grads = grad_fn(data, label)
            optimizer(grads)
                
            return loss,out
        
        def forward_fn(data, label):
            out, _, _ = self.clf(data)
            loss_fn = nn.CrossEntropyLoss()  
            loss = loss_fn(out, label)
            return loss, out
        
        grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

        for batch_idx, (x, y, _, _) in enumerate(loader_tr):
            x_np = x.asnumpy()
            x_reshaped = np.squeeze(x_np, axis=1)
            x = Tensor(x_reshaped)
            
            if soft_target: 
                y = y.astype(mindspore.float32)
            else:
                y = y.astype(mindspore.int32)

            loss,out = train_step(x, y)
            loss_list.append(loss.item())
            
            if soft_target:
                pred_compare = out.max(axis=1, return_indices=True)[1]
                y_compare = y.max(axis=1, return_indices=True)[1]
                right_count_list.append((pred_compare == y_compare).sum().item())
                samples_per_batch.append(len(y))
            else:
                pred = out.max(axis=1, return_indices=True)[1]
                right_count_list.append((pred == y).sum().item())
                samples_per_batch.append(len(y))
            
            iter_time = self.timer.since_last_check()
            if log_show:
                if (batch_idx + 1) % iter_out == 0:
                    log_dict = dict(
                        mode=clf_name,  # 训练模式
                        epoch=self.epoch,  # 当前是第几个epoch
                        iter=batch_idx + 1,  # 当前是第几个iter
                        lr=get_lr(clf_group['optimizer'],self.epoch),  # 获取当前optimizer的学习率
                        time=iter_time,  # 当前iter消耗的时间
                        acc=1.0 * np.sum(right_count_list[-iter_out:]) / np.sum(samples_per_batch[-iter_out:]),
                        loss=np.sum(loss_list[-iter_out:])
                    )
                    self.TextLogger.log(
                        log_dict=log_dict,
                        iters_per_epoch=len(loader_tr),
                        iter_count=self.epoch * len(loader_tr) + batch_idx,
                        max_iters=self.args.n_epoch * len(loader_tr),  # 一个active round总共的epoch数量
                        interval=iter_out  # 多少个iter进行一次log
                    )
            
            

    def augment_optimizer_label(self, metric='loss', split_guide='val', split_train='train'):
        # 若不优化，则读取并返回一个固定值
        if self.args.aug_strength_lab is not None:
            if self.args.aug_lab_strength_mode == 'sample':
                return mindspore.ops.ones([len(self.dataset.DATA_INFOS[split_train])]) * self.args.aug_strength_lab
            elif self.args.aug_lab_strength_mode == 'class':
                return mindspore.ops.ones([len(self.dataset.CLASSES)]) * self.args.aug_strength_lab
            elif self.args.aug_lab_strength_mode == 'all':
                return self.args.aug_strength_lab
            else:
                raise NotImplementedError
        # 需要把magnitude转为能输入的value/参数列表
        # 有以下几种不同的模式:
        # 1.优化关于整个数据集的一个整体变量
        # 2.对于每个类优化一个整体变量
        # 3.对于每个样本优化一个变量
        # optimize a strength before training
        # 先进行没有任何增强的初步训练
            
        self.clf.set_train(True)

        dataset_tr_init = GetHandler(self.dataset, split_guide, self.dataset.default_train_transform)
        
        loader_tr_init = mindspore.dataset.GeneratorDataset(dataset_tr_init, column_names=["data_x","data_y","data_no","data_idx"], shuffle=True, 
                                    num_parallel_workers=self.args.num_workers)
        loader_tr_init = loader_tr_init.batch(self.args.batch_size)

        while self.epoch < self.args.n_epoch:
            self.timer.since_last_check()
            self._train(loader_tr_init, {'clf': self.clf, 'optimizer': self.optimizer, 'scheduler': self.scheduler})
            if self.epoch % self.args.save_freq == 0 and self.epoch > 0:
                pass
                # self.save()
            self.epoch += 1
        self.epoch = 0

        # 据此优化增强强度
        sample_strength_matrix = mindspore.ops.zeros([self.args.num_strength_bins_lab,
                                              len(self.dataset.DATA_INFOS[split_train])])
        for strength in range(1, self.args.num_strength_bins_lab+1):
            # 这里还没有加入强度，请小心处理
            dataset = GetHandler(self.dataset, split_train, self.dataset.default_val_transform, repeat_times=1,
                                 single_aug_times=2, mix_aug_times=1,
                                 aug_mode=self.args.aug_lab_training_mode,
                                 strength_mode="sample",
                                 num_strength_bins=self.args.num_strength_bins_lab,
                                 args_list=mindspore.ops.ones([len(self.dataset.DATA_INFOS[split_train])]) * strength,
                                 ablation_aug_type=self.args.aug_type_lab, ablation_mix_type=self.args.mix_type_lab)
            scores = self.predict(self.clf, dataset, metric=metric, log_show=False, split_info='train_optim')
            sample_strength_matrix[strength - 1, :] = \
                mindspore.ops.sum(scores.reshape([len(self.dataset.DATA_INFOS[split_train]), -1]), dim=1)
            
        # per_sample_magnitude
        if self.args.aug_lab_strength_mode == 'sample':
            return mindspore.ops.argmin(sample_strength_matrix, axis=0) + 1
        # per_class_magnitude
        elif self.args.aug_lab_strength_mode == 'class':
            # 首先获得主类下标

            idx_to_class = mindspore.Tensor([int(elem['gt_label']) for elem in self.dataset.DATA_INFOS[split_train]])
            class_sample_strength_matrix = mindspore.ops.zeros(self.args.num_strength_bins_lab,
                                                       len(self.dataset.CLASSES))
            for i in range(len(self.dataset.CLASSES)):
                class_sample_strength_matrix[:, i] += mindspore.ops.sum(sample_strength_matrix[:, idx_to_class == i], dim=1)
            return mindspore.ops.argmin(class_sample_strength_matrix, axis=0) + 1
        else:
            new_matrix = mindspore.ops.sum(sample_strength_matrix, dim=1)
            return mindspore.ops.argmin(new_matrix).item() + 1
        # global_magnitude

    def augment_optimizer_unlab(self, metric='entropy'):
        if self.args.aug_strength_ulb is not None:
            if self.args.aug_ulb_strength_mode == 'sample':
                return mindspore.ops.ones([len(self.dataset.DATA_INFOS['train_u'])]) * self.args.aug_strength_ulb
            elif self.args.aug_ulb_strength_mode == 'class':
                return mindspore.ops.ones([len(self.dataset.CLASSES)]) * self.args.aug_strength_ulb
            elif self.args.aug_ulb_strength_mode == 'all':
                return self.args.aug_strength_ulb
            else:
                raise NotImplementedError
        # 据此优化增强强度
        sample_strength_matrix = mindspore.ops.zeros([self.args.num_strength_bins_ulb,
                                              len(self.dataset.DATA_INFOS['train_u'])])
        
        
        for strength in range(1, self.args.num_strength_bins_ulb + 1):
            dataset = GetHandler(self.dataset, 'train_u', self.dataset.default_val_transform, repeat_times=1,
                                 single_aug_times=2, mix_aug_times=1,
                                 aug_mode=self.args.aug_ulb_evaluation_mode,
                                 strength_mode="sample",
                                 num_strength_bins=self.args.num_strength_bins_ulb,
                                #  args_list=torch.ones([len(self.dataset.DATA_INFOS['train_u'])]) * strength,
                                 args_list=mindspore.ops.ones([len(self.dataset.DATA_INFOS['train_u'])]) * strength,
                                 ablation_aug_type=self.args.aug_type_ulb, ablation_mix_type=self.args.mix_type_ulb)
            scores = self.predict(self.clf, dataset, metric=metric, log_show=False, split_info='train_u_optim')

            sample_strength_matrix[strength - 1, :] = \
                mindspore.ops.min(scores.reshape([len(self.dataset.DATA_INFOS['train_u']), -1]), axis=1)[0]

        # per_sample_magnitude
        if self.args.aug_ulb_strength_mode == 'sample':
            return mindspore.ops.argmax(sample_strength_matrix, dim=0) + 1
        
        # per_class_magnitude
        elif self.args.aug_ulb_strength_mode == 'class':
            # 首先获得主类下标
            # 需要先获得无标注样本的伪标注
            self.get_vt_label()
            idx_to_class = Tensor([int(elem['vt_label']) for elem in self.dataset.DATA_INFOS['train_u']])
            class_sample_strength_matrix = mindspore.ops.zeros(self.args.num_strength_bins_ulb, len(self.dataset.CLASSES))
            for i in range(len(self.dataset.CLASSES)):
                class_sample_strength_matrix[:, i] += mindspore.ops.sum(sample_strength_matrix[:, idx_to_class == i], dim=1)
            return mindspore.ops.argmax(class_sample_strength_matrix, dim=0) + 1
        else:
            new_matrix = mindspore.ops.sum(sample_strength_matrix, dim=1)
            return mindspore.ops.argmax(new_matrix).item() + 1

    def train(self, strength_mode=None, args_list=None):
        # 如果需要进行有优化的增强，先进行增强的判断
        
        if args_list is None:
            if self.args.aug_lab_training_mode in ['StrengthGuidedAugment', 'RandAugment']:
                args_list = self.augment_optimizer_label()
        if strength_mode is None:
            strength_mode = self.args.aug_lab_strength_mode

        # 正式开始训练
        self.logger.info('Start running, host: %s, work_dir: %s',
                         f'{getuser()}@{gethostname()}', self.args.work_dir)
        self.logger.info('max: %d epochs', self.args.n_epoch)
        self.init_clf()
        
        dataset_tr = GetHandler(self.dataset, 'train', self.dataset.default_train_transform, 1,
                                self.args.aug_ratio_lab, self.args.mix_ratio_lab, self.args.aug_lab_training_mode,
                                strength_mode=strength_mode, args_list=args_list,
                                ablation_aug_type=self.args.aug_type_lab, ablation_mix_type=self.args.mix_type_lab)
        
        loader_tr = mindspore.dataset.GeneratorDataset(dataset_tr, shuffle=False,  column_names=["data_x","data_y","data_no","data_idx"],
                                    num_parallel_workers=self.args.num_workers)
                
        def reshape_data_x(data_x):
        # 如果 data_x 形状为 (3, 32, 32)，则添加一个维度变成 (1, 3, 32, 32)        
            if len(data_x.shape) == 3:
                C, H, W = data_x.shape
                return data_x.reshape(1, C, H, W)
            else:
                return data_x  # 如果形状已经是 (1, 3, 32, 32)，则保持不变

        # 定义一个处理函数来应用于 GeneratorDataset 中的每个样本
        def map_func(data_x, data_y, data_no, data_idx):
            data_x = reshape_data_x(data_x)
            return data_x, data_y, data_no, data_idx

        # 使用 map 函数将处理函数应用到 GeneratorDataset 中
        loader_tr = loader_tr.map(input_columns=["data_x", "data_y", "data_no", "data_idx"],
                                operations=map_func,
                                num_parallel_workers=self.args.num_workers)

        loader_tr = loader_tr.batch(self.args.batch_size, num_parallel_workers=self.args.num_workers)
        while self.epoch < self.args.n_epoch:
            self.timer.since_last_check()
            self._train(loader_tr, {'clf': self.clf, 'optimizer': self.optimizer, 'scheduler': self.scheduler},
                        soft_target=True if (self.args.mix_ratio_lab > 0) else False)
            if self.epoch % self.args.save_freq == 0 and self.epoch > 0:
                pass
                # self.save()
            self.epoch += 1
        self.epoch = 0
        # self.save()

    def run(self):
        while self.cycle < self.args.n_cycle:
            # 每个cycle中参数的记录位置
            active_path = os.path.join(self.args.work_dir, f'active_cycle_{self.cycle}')
            os.makedirs(active_path, mode=0o777, exist_ok=True)
            num_labels = len(self.dataset.DATA_INFOS['train'])
            self.logger.info(f'Active Round {self.cycle} with {num_labels} labeled instances')
            active_meta_log_dict = dict(
                mode='active_meta',
                cycle=self.cycle,
                num_labels=num_labels
            )
            self.TextLogger._dump_log(active_meta_log_dict)
            if not self.args.updating:
                self.init_clf()
            self.init_cycle_info()
            # 如果有有标注数据增强，在这一步需要优化之
            
            self.train()
        

            
            dataset_val = GetHandler(self.dataset, 'val', self.dataset.default_val_transform)
            dataset_test = GetHandler(self.dataset, 'test', self.dataset.default_val_transform)
            self.acc_val_list.append(self.predict(self.clf, dataset_val, split_info='val'))
            self.acc_test_list.append(self.predict(self.clf, dataset_test, split_info='test'))
            self.num_labels_list.append(num_labels)
            # 新版代码的增强点在这儿, 修改update中的内容
            # 如果有无标注数据增强，在这一步需要优化之
            self.update(self.args.num_query)  # Update the labeled pool according to the current model
            # 更新info中的关于查询的内容
            for idx in self.dataset.QUERIED_HISTORY[-1]:
                self.cycle_info[idx]['queried'] = 1
            self.cycle += 1
        self.record_test_accuracy()

    def predict(self, clf, dataset: Handler, metric='accuracy',
                topk=None, n_drop=None, thrs=None, dropout_split=False, log_show=True, split_info='train'):
        # For both evaluation and informative metric based on probabilistic outputs
        # Allowed split: train, train_full, val, test
        # Allowed metrics: accuracy, precision, recall, f1_score, support
        # The above metrics return a scalar
        # Allowed informative metrics: entropy, lc, margin
        # The above metrics return a vector of length N(The number of data points)
        # If in dropout split mode, The above metrics return a tensor of size [n_drop, N, C]

        loader = mindspore.dataset.GeneratorDataset(dataset, shuffle=False,column_names=["data_x","data_y","data_no","data_idx"], 
                                    num_parallel_workers=self.args.num_workers)

        isCell = isinstance(clf, nn.Cell)
        if isCell:
            clf.set_train(False)
        if n_drop is None:
            n_drop = 1
        if topk is None:
            topk = 1
        if thrs is None:
            thrs = 0.
        if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'support', 'loss']:
            # Evaluation Metric
            self.logger.info(f"Calculating Performance with {metric}...")
            pred = []
            target = []
            for x, y, _, idxs in track_iter_progress(loader):
                if isCell:
                    if len(x.shape) == 3 : # (3, 32, 32):
                        x = mindspore.ops.ExpandDims()(x, 0) 
                    out, _, _ = clf(x)
                else:
                    out = clf(x)
                prob = mindspore.ops.softmax(out, axis=1)
                pred.append(Tensor(prob))

                target.append(y.astype(mindspore.float32))
            
            if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'support']:
                # 注意这里不能有mixing样本, 否则出错
                pred = mindspore.ops.cat(pred)
                target = mindspore.ops.stack(target, axis=0)

                if metric == 'accuracy':
                    result = accuracy(pred, target, topk, thrs)
                elif metric == 'precision':
                    result = precision(pred, target, thrs=thrs)
                elif metric == 'recall':
                    result = recall(pred, target, thrs=thrs)
                elif metric == 'f1_score':
                    result = f1_score(pred, target, thrs=thrs)
                elif metric == 'support':
                    result = support(pred, target)
                else:
                    raise Exception(f"Metric {metric} not implemented!")
                if len(result) == 1:
                    result = result.item()
                else:
                    result = result.numpy().tolist()
                if log_show:
                    log_dict = dict(mode=split_info, cycle=self.cycle)
                    log_dict[metric] = result
                    self.TextLogger.log(log_dict)
            # 如果需逐个样本算loss
            else:
                
                pred = mindspore.ops.cat(pred)
                target = mindspore.ops.cat(target,axis=0)
                target = target.reshape(pred.shape)



                result = mindspore.ops.cross_entropy(pred, target, reduction='none')

        else:  # Informative Metric
            self.logger.info(f"Calculating Informativeness with {metric}...")
            if isCell:
                clf.set_train(True)
            if dropout_split is False:
                pred = []
                for i in range(n_drop):
                    self.logger.info('n_drop {}/{}'.format(i + 1, n_drop))
                    for batch_idx, (x, _, _, _) in enumerate(track_iter_progress(loader)):
            
                        if isCell:
                            if len(x.shape) == 3 : # (3, 32, 32):
                                x = mindspore.ops.ExpandDims()(x, 0) 
                            out, _, _ = clf(x)
                        else:
                            out = clf(x)
                        if i == 0:
                            pred.append(mindspore.ops.softmax(out, axis=1))
                        else:
                            pred[batch_idx] += mindspore.ops.softmax(out, axis=1)
                        

                pred = mindspore.ops.cat(pred)

                pred /= n_drop
                if metric == 'entropy':
                    log_pred = mindspore.ops.log(pred)
                    
                    # the larger the more uncertain
                    result = - (pred * log_pred).sum(1)
                elif metric == 'lc':
                    # the smaller the more uncertain
                    result = 1.0 - pred.max(axis=1, return_indices=True)[0]
                elif metric == 'margin':
                    # the smaller the more uncertain
                    pred_sorted, _ = pred.sort(descending=True)
                    result = 1.0 - (pred_sorted[:, 0] - pred_sorted[:, 1])
                elif metric == 'prob':
                    result = pred
                else:
                    raise Exception(f"Metric {metric} not implemented!")
            else:
                print("No metric will be used in dropout split mode!")
                data_length = len(self.dataset)
                result = mindspore.ops.zeros([n_drop, data_length, len(self.dataset.CLASSES)])

                for i in range(n_drop):
                    self.logger.info('n_drop {}/{}'.format(i + 1, n_drop))

                    for x, _, _, idxs in track_iter_progress(loader):
                        
                        if isCell:
                            if len(x.shape) == 3 : # (3, 32, 32):
                                x = mindspore.ops.ExpandDims()(x, 0) 
                            out, _, _ = clf(x)
                        else:
                            out = clf(x)
                        result[i][idxs] += mindspore.ops.softmax(out, axis=1)

        # n_drops ignored
        return result
        # back to train split as the default split


    def get_embedding(self, clf, dataset, embed_type='default'):
        # type can ba chosen from default or grad
        # 此处不再对标注集进行筛选，而是全部使用
        loader = mindspore.dataset.GeneratorDataset(dataset, shuffle=False,  column_names=["data_x","data_y","data_no","data_idx"],
                            num_parallel_workers=self.args.num_workers)

        clf.set_train(False)
        self.logger.info(f"Extracting embedding of type {embed_type}...")
        embdim = self.get_embedding_dim()
        nlabs = len(self.dataset.CLASSES)
        if embed_type == 'default':
            embedding = []
            for x, _, _, idxs in track_iter_progress(loader):
                if len(x.shape) == 3 : # (3, 32, 32):
                    x = mindspore.ops.ExpandDims()(x, 0) 
                _, e1, _ = clf(x)
                embedding.append(e1)
            embedding = mindspore.ops.cat(embedding)

        elif embed_type == 'grad':
            data_length = len(dataset)
            embedding = np.zeros([data_length, embdim * nlabs])
            
            for batch_idx, (x, y, _, idxs) in enumerate(track_iter_progress(loader)):    
                
                if len(x.shape) == 3 : # (3, 32, 32):
                    x = mindspore.ops.ExpandDims()(x, 0) 
                cout, e, _ = clf(x)
                out = e.asnumpy()
                batchProbs = mindspore.ops.softmax(cout, axis=1).asnumpy()
                maxInds = np.argmax(batchProbs, 1)
                
                idxs = mindspore.ops.ExpandDims()(idxs, 0)
                
                for j in range(1):
                    for c in range(nlabs):
                        if c == maxInds[j]:
                            update_value = deepcopy(out[j]) * (1 - batchProbs[j][c])                                
                        else:
                            update_value = deepcopy(out[j]) * (-1 * batchProbs[j][c])
                        embedding[idxs[j]][embdim * c: embdim * (c + 1)] = update_value
            return Tensor(embedding)
        else:
            raise Exception(f'Embedding of type {embed_type} not implemented!')
        return embedding

    def get_embedding_dim(self) -> int:
        dataset = GetHandler(self.dataset, 'train', self.dataset.default_val_transform, 1)
        # 此处不再对标注集进行筛选，而是全部使用
        loader = mindspore.dataset.GeneratorDataset(dataset, shuffle=False,  column_names=["data_x","data_y","data_no","data_idx"],
                                    num_parallel_workers=self.args.num_workers)
        loader = loader.batch(self.args.batch_size)

        self.clf.set_train(False)

        for x, _, _, _ in loader:
            squeeze = mindspore.ops.Squeeze(1)
            x = squeeze(x) # (50,3,32,32)
            if len(x.shape) == 3 : # (3, 32, 32):
                x = mindspore.ops.ExpandDims()(x, 0) 
            _, e1, _ = self.clf(x)
            return e1.shape[1]

    def get_vt_label(self):
        dataset = GetHandler(self.dataset, 'train_u', self.dataset.default_val_transform)
        loader = mindspore.dataset.GeneratorDataset(dataset, shuffle=False,  column_names=["data_x","data_y","data_no","data_idx"],
                            num_parallel_workers=self.args.num_workers)
        
        self.logger.info(f"Calculating virtual labels for with unlabeled samples.")
        pred = []
        for x, _, _, _ in track_iter_progress(loader):
            if isinstance(self.clf, nn.Cell):
                if len(x.shape) == 3 : # (3, 32, 32):
                    x = mindspore.ops.ExpandDims()(x, 0) 
                out, _, _ = self.clf(x)
            else:
                out = self.clf(x)
            pred_elem = mindspore.ops.argmax(out, axis=1)
            pred.append(pred_elem)

        pred = mindspore.ops.cat(pred)
        for idx, vt_label in enumerate(pred):
            self.dataset.DATA_INFOS['train_u'][idx]['vt_label'] = int(vt_label)

    def save(self):
        """Save the current model parameters."""
        model_out_path = '/home/jovyan/CAMPAL-mindspore/savemodel'
        save_target = model_out_path + '/trainmodel.ckpt'

        mindspore.save_checkpoint(self.clf, save_target)

        self.logger.info('==> save model to {}'.format(save_target))

    def record_test_accuracy(self):
        file_name = os.path.join(self.args.work_dir, 'accuracy.csv')
        header = ['num_labels', 'accuracy']
        with open(file_name, 'w', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(header)
            for i, acc in enumerate(self.acc_test_list):
                f_csv.writerow([(i + 1) * self.args.num_query, acc])
