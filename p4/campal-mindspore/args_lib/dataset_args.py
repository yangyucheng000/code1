dataset_dict = {
    'cifar10': {'model': 'resnet18_cifar', 'dataset': 'cifar10', 'num-init-labels': 100, 'num-query': 100,
                'subset': 10000},
    'cifar100': {'model': 'resnet18_cifar', 'dataset': 'cifar100', 'num-init-labels': 1000, 'num-query': 1000,
                 'subset': 10000},
    'svhn': {'model': 'resnet18_cifar', 'dataset': 'svhn', 'num-init-labels': 100, 'num-query': 100, 'subset': 10000},
    'caltech101': {'model': 'resnet18', 'dataset': 'caltech101', 'num-init-labels': 102, 'num-query': 102,
                   'subset': 10200},
    'caltech256': {'model': 'resnet18', 'dataset': 'caltech256', 'num-init-labels': 257, 'num-query': 257,
                   'subset': 25700},
    'tinyimagenet': {'model': 'resnet18', 'dataset': 'tinyimagenet', 'num-init-labels': 200, 'num-query': 200,
                     'subset': 20000, 'aug-trials': 5},
    'imagenet': {'model': 'resnet18', 'dataset': 'sampledimagenet', 'num-init-labels': 1000, 'num-query': 1000,
                 'subset': 50000, 'aug-trials': 2, 'aug-lab': 2, 'aug-ulb': 1}
}
