default_dict = {
    'model': 'resnet18_cifar',
    'dataset': 'cifar10',
    'strategy': 'EntropySampling',
    'num-init-labels': 100,
    'n-cycle': 20,
    'num-query': 100,
    'n-epoch': 300,
    'batch-size': 50,
    'lr': 0.1,
    'momentum': 0.9,
    'weight-decay': 0.0005,
    'milestones': [120, 180, 240],
    'subset': 10000,
    'aug-lab': 2,
    'aug-ulb': 4,
    'ablation-aug-type': 'mixup',
    'pace': 0.
}
