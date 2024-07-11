import datetime
import os
import time
import uuid
import random

import numpy as np
from datasets.builder import DATASETS
from architectures.builder import MODELS
from query_strategies.builder import STRATEGIES
from utils.config import parse_commandline_args
from utils.logger import get_logger
from utils.collect_env import collect_env
from utils.timer import Timer
import matplotlib.pyplot as plt

import mindspore


def set_seed(seed=0):
    """If the seed is specified, the process will be deterministic.

    :param seed: the seed you wanna set
    :return: None

    """
    random.seed(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)



def run(config: dict = None):
    uid = str(uuid.uuid1().hex)[:8]
    # resumed = (config.work_dir is not None)
    if config.work_dir is None:
        config.work_dir = os.path.join('tasks',
                                       '{}_{}_{}_no_query_{}_'
                                       'AugTypeLab_{}_AugTypeUlb_{}_Setting_{}_{}_{}'.format(
                                           config.model,
                                           config.dataset,
                                           config.strategy,
                                           config.num_query,
                                           config.aug_lab_training_mode + '_' + config.aug_lab_strength_mode,
                                           config.aug_ulb_evaluation_mode + '_' + config.aug_ulb_strength_mode,
                                           config.setting_name,
                                           datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"),
                                           uid))
    os.makedirs(config.work_dir, mode=0o777, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    config.timestamp = timestamp
    log_file = os.path.join(config.work_dir, f'{timestamp}.log')
    logger = get_logger(name='DAL', log_file=log_file)
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    # set seed
    if config.seed is not None:
        set_seed(config.seed)  # To make the process deterministic

    # load dataset
    dataset = DATASETS.build(
        dict(type=config.dataset, initial_size=config.num_init_labels))

    # start experiment
    n_pool = len(dataset.DATA_INFOS['train_full'])
    n_eval = len(dataset.DATA_INFOS['val'])
    n_test = len(dataset.DATA_INFOS['test'])
    logger.info('cardinality of initial labeled pool: {}'.format(config.num_init_labels))
    logger.info('cardinality of initial unlabeled pool: {}'.format(n_pool - config.num_init_labels))
    logger.info('cardinality of initial evaluation pool: {}'.format(n_eval))
    logger.info('cardinality of initial test pool: {}'.format(n_test))

    

    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    mindspore.set_context(pynative_synchronize=False)
    mindspore.dataset.config.set_num_parallel_workers(8)
    
    # print('当前线程数：',mindspore.dataset.config.get_num_parallel_workers()) 
    
    
    # load network
    net = MODELS.build(dict(type=config.model))
    strategy = STRATEGIES.build(dict(type=config.strategy,
                                     dataset=dataset,
                                     net=net, args=config,
                                     logger=logger, timestamp=timestamp))

    # print info
    logger.info('Dataset: {}'.format(config.dataset))
    logger.info('Seed: {}'.format(config.seed))
    logger.info('Strategy: {}'.format(type(strategy).__name__))

    if config.load_path is not None:
        strategy.clf.load_state_dict(mindspore.load(config.load_path))
        logger.info(f'Get pretrained parameters from {config.load_path}')

    strategy.run()

    # plot acc - label_num curve
    plt.figure()
    plt.plot(strategy.num_labels_list, strategy.acc_test_list, 'r-*', lw=1, ms=5)
    plt.savefig(os.path.join(config.work_dir, 'acc_num_labels.png'))
    plt.clf()


if __name__ == '__main__':
    # MindSpore会自动根据系统的CPU核心数进行并行计算，无需手动设置线程数。
    with Timer():
        config = parse_commandline_args()
        run(config)
