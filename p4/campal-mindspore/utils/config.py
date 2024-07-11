import argparse

__all__ = ['parse_commandline_args']


def create_parser():
    """Get the args from the command line"""
    parser = argparse.ArgumentParser(description='Deep active learning args --PyTorch ')

    # 与存储相关的必要信息
    # 存储的文件夹目录
    parser.add_argument('--work-dir', default=None, type=str, help='the dir to save logs and models')
    # 每隔多少epoch存储一次模型
    parser.add_argument('--save-freq', default=100, type=int, metavar='EPOCHS',
                        help='checkpoint frequency(default: 100)')

    # 常规深度模型训练配置
    # 模型架构
    parser.add_argument('--model', default='resnet18', metavar='MODEL')
    # 训练的数据集
    parser.add_argument('--dataset', type=str, default='cifar10', metavar='DATASET',
                        help='The name of the used dataset(default: cifar10)')
    # 是否载入预训练模型
    parser.add_argument('--load-path', type=str, default=None, help='which pth file to preload')
    # Setting的名称，不参与模型实际训练流程，仅作标注用
    parser.add_argument('--setting-name', type=str, default=None, help='The setting name')

    # 主动学习策略选择
    # 策略名称
    parser.add_argument('--strategy', type=str, default='EntropySampling',
                        help='which sampling strategy to choose')
    # 初始标注池样本数量
    parser.add_argument('--num-init-labels', default=100, type=int,
                        metavar='N', help='number of initial labeled samples(default: 100)')
    # 训练的cycle数量
    parser.add_argument('--n-cycle', default=2, type=int,
                        metavar='N', help='number of query rounds(default: 10)')
    # 每个cycle查询的样本数量
    parser.add_argument('--num-query', default=100, type=int,
                        metavar='N', help='number of query samples per epoch(default: 100)')
    # 为了节省计算时间，对无标注池进行随机采样，该数字表示每次查询时采样的无标注样本数量
    # 若数字足够大或者为None则全部使用
    parser.add_argument('--subset', default=10000, type=int,
                        metavar='N', help='the size of the unlabeled pool to query, subsampling')
    # 若添加此选项，则基于上一次cycle更新模型；否则每次都是重新训练
    parser.add_argument('--updating', action='store_true', help='Whether to use updating or retraining')
    # 每一轮训练的epoch数量
    parser.add_argument('--n-epoch', default=2, type=int, metavar='N',
                        help='number of total training epochs(default: 100)')

    # 数据集初始配置分布相关
    # 这里添加和不平衡度相关的配置，目前暂未决定
    # 用到的数据集的不平衡模式
    parser.add_argument('--dataset-imbalance-mode', default=None, type=str, help='Imbalance mode for the dataset used')
    # 初始集的不平衡模型
    parser.add_argument('--init-imbalance-mode', default=None, type=str,
                        help='Imbalance mode for the initial labeled pool')

    # 其他超参配置
    parser.add_argument('--batch-size', type=int, default=50, metavar='BATCH_SIZE',
                        help='Batch size in both train and test phase(default: 64)')
    parser.add_argument('--num-workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='max learning rate (default: 0.1)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay (default: 0.0001)')
    parser.add_argument('--milestones', default=[100, 180, 240], type=int, nargs='+',
                        help='milestones of learning scheduler to use '
                             '(default: [100, 180, 240])')
    parser.add_argument('--seed', default=None, type=int, metavar='SEED', help='Random seed (default: None)')
    # console以及json中的输出频率
    parser.add_argument('--out-iter-freq', default=10, type=int)

    # 关于有标注数据增强的配置
    # 有标注数据增强是否启动，一般作为flag标识用
    parser.add_argument('--aug-lab-on', action='store_true', help='whether to use labeled augmentation')
    # 原数据集的复制次数
    parser.add_argument('--duplicate-ratio', default=1, type=int, help='duplicate ratio of the labeled pool')
    # 有标注池单样本增强数量
    parser.add_argument('--aug-ratio-lab', default=1, type=int, help='single-image augmentation ratio')
    # 有标注池混合样本增强数量
    parser.add_argument('--mix-ratio-lab', default=1, type=int, help='image mixing augmentation ratio')

    # 有标注池单样本增强类型
    # 1. 可以是某种增强类型，或者多种手段的列表，比如autocontrast，rotation等，专门用于ablation
    # 2. 可以是某种常用自动增强手段，比如autoaugment，randaugment，trivialaugment等，也专门用于ablation
    # 3. 可以是我们提出的方法，名字待定
    # 4. 可以是某种默认方法，比如random，从增强池中随机采样
    parser.add_argument('--aug-lab-training-mode', default='StrengthGuidedAugment', type=str,
                        help='how augmentations interfere with training 0. StrengthGuidedAugment;'
                             '1. RandAugment; 2. AutoAugment; 3. TrivialAugmentWide',)
    parser.add_argument('--aug-lab-strength-mode', default='all', type=str,
                        help='strength is optimized 0. default mode: none'
                             '1. globally: all; 2. per class: class; 3. per sample: sample')
    parser.add_argument('--num-strength-bins-lab', default=4, type=int,
                        help='The number of strengths to divide')
    # Ablation专用参数，平时为None，给定值则不再自动优化
    # 单样本增强类型
    parser.add_argument('--aug-type-lab', default=None, type=str, help='the augmentation type used')
    # 混合样本增强类型
    parser.add_argument('--mix-type-lab', default=None, type=str,
                        help='the mixing types used for ablation purpose only')
    # 有标注池混合样本增强强度，在自动增强中可以不指定（指定为None）
    parser.add_argument('--aug-strength-lab', default=None, type=int,
                        help='augmentation magnitude for labeled pool, '
                             'When it is None, meaning automatic optimization for magnitudes; '
                             'When it is a given number, it is fixed, no optimization.'
                             'For ablation purpose only.')

    # 关于无标注数据增强的配置
    # 无标注数据增强是否启动，一般作为flag标识用
    parser.add_argument('--aug-ulb-on', action='store_true', help='whether to use labeled augmentation')
    # 无标注池单样本增强数量
    parser.add_argument('--aug-ratio-ulb', default=1, type=int, help='single-image augmentation ratio')
    # 无标注池样本混合增强数量
    parser.add_argument('--mix-ratio-ulb', default=1, type=int, help='image mixing augmentation ratio')

    # 无标注池单样本增强类型
    # 1. 可以是某种增强类型，或者多种手段的列表，比如autocontrast，rotation等，专门用于ablation
    # 2. 可以是我们提出的方法，名字待定（无标注样本不存在自动数据增强）
    # 3. 可以是某种默认方法，比如random，从增强池中随机采样
    parser.add_argument('--aug-ulb-evaluation-mode', default='StrengthGuidedAugment', type=str,
                        help='how augmentations interfere with training 0. StrengthGuidedAugment;'
                             '1. RandAugment; 2. AutoAugment; 3. TrivialAugmentWide', )
    parser.add_argument('--aug-ulb-strength-mode', default='all', type=str,
                        help='strength is optimized 0. default mode: none'
                             '1. globally: all; 2. per class: class; 3. per sample: sample')
    parser.add_argument('--num-strength-bins-ulb', default=4, type=int,
                        help='The number of strengths to divide')
    parser.add_argument('--aug-metric-ulb', default='normal',
                        type=str, help='Only for augmentation-based metrics, including:')

    # Ablation专用参数，平时为None，给定值则不再自动优化
    # 单样本增强类型
    parser.add_argument('--aug-type-ulb', default=None, type=str, help='the augmentation types used, for ablation only')
    # 混合样本增强类型
    parser.add_argument('--mix-type-ulb', default=None, type=str, help='the mixing types used, for ablation only')
    # 无标注池混合样本增强强度，在自动增强中可以不指定（指定为None）
    parser.add_argument('--aug-strength-ulb', default=None, type=int,
                        help='augmentation magnitude for labeled pool, '
                             'When it is None, meaning automatic optimization for magnitudes; '
                             'When it is a given number, it is fixed, no optimization.'
                             'For ablation purpose only.')

    return parser


def parse_commandline_args():
    """Returns the args from the command line"""
    return create_parser().parse_args()
