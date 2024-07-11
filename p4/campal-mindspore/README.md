# CAMPAL-mindspore

This is a Mindspore implementation of our ICML 2023 paper [CAMPAL](https://openreview.net/forum?id=DXWm3vnG6P).

[Pytorch implementation](https://github.com/jnzju/CAMPAL)

- Title: Active Learning with Controllable Augmentation-induced Acquisition
- Authors: Jianan Yang, Haobo Wang, Sai Wu, Gang Chen, Junbo Zhao

```json
@article{yang2023active,
  title={Active Learning with Controllable Augmentation Induced Acquisition},
  author={Yang, Jianan and Wang, Haobo and Wu, Sai and Chen, Gang and Zhao, Junbo},
  year={2023}
}
```

## Environment Setup

Please refer to the project [source address](https://github.com/jnzju/CAMPAL) for the dataset and parameter configuration.

Recommended Hardware Environment(Ascend/GPU/CPU) : GPU cuda11.6

Recommended Software Environment:

- MindSpore version : 2.2.10
- Python version: 3.9.18
- OS platform and distribution : Linux Ubuntu 20.04 x86_64 GNU/Linux
- GCC/Compiler version (if compiled from source): 8.5.0
- Additional libraries: pyyaml, scikit learn, matplotlib

## Run CAMPAL in mindspore

You can just run the `main.py` or use the bash example that we provide in `./sh`

## Evaluation

Comparison of evaluation under Mindspore and PyTorch architectures

1. Dataset: Cifar10, epoch = 50

|         |            | Strategy        | NL = 500   | NL=1000   | NL=1500   | NL=2000   |
| ------- | ---------- | --------------- | ---------- | --------- | --------- | --------- |
|         |            | LeastConfidence | 40.72      | 55.19     | 54.43     | 65.45     |
|         | Pytorch    | EntroySampling  | 40.25      | 47.12     | 55.85     | 64.52     |
|         |            | BadgeSampling   | 39.15      | 53.06     | 55.99     | 61.07     |
| Cifar10 | ---------- | --------------- | ---------- | --------- | --------- | --------- |
|         |            | LeastConfidence | 40.44      | 41.22     | 54.76     | 57.82     |
|         | Mindspore  | EntroySampling  | 35.11      | 42.95     | 51.93     | 59.1      |
|         |            | BadgeSampling   | 38.29      | 43.57     | 52.36     | 61.5      |

2. Dataset: SVHN, epoch = 30

|      |            | Strategy        | NL = 500   | NL=1000   | NL=1500   | NL=2000   |
| ---- | ---------- | --------------- | ---------- | --------- | --------- | --------- |
|      |            | LeastConfidence | 25.49      | 71.84     | 80.88     | 83.38     |
|      | Pytorch    | EntroySampling  | 18.47      | 20.79     | 81.12     | 81.55     |
|      |            | BadgeSampling   | 39.32      | 22.28     | 76.25     | 82.17     |
| SVHN | ---------- | --------------- | ---------- | --------- | --------- | --------- |
|      |            | LeastConfidence | 24.09      | 58.75     | 75.36     | 77.51     |
|      | Mindspore  | EntroySampling  | 18.78      | 60.20     | 70.84     | 82.89     |
|      |            | BadgeSampling   | 29.77      | 36.13     | 74.63     | 79.26     |
