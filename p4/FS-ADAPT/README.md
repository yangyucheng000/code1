Few-Shot Time-Series Anomaly Detection with Unsupervised Domain Adaptation

Anomaly detection for time-series data is crucial in the management of systems for streaming applications, computational services, and cloud platforms. The majority of current few-shot learning (FSL) approaches are supposed to discover the remarkably low fraction of anomaly samples in a large number of time-series samples. Furthermore, due to the tremendous effort required to label data, most time-series datasets lack data labels, necessitating unsupervised domain adaptation (UDA) methods. Therefore, time-series anomaly detection is a problem that combines the aforementioned two difficulties, termed FS-UDA. To solve the problem, we propose a Few-Shot time-series Anomaly Detection framework with unsupervised domAin adaPTation (FS-ADAPT), which consists of two modules: a dueling triplet network to address the constraints of unsupervised target information, and an incremental adaptation module for addressing the limitations of few anomaly samples in an online scenario. The dueling triplet network is adversarially trained with augmented data and unlabeled target samples to learn a classifier. The incremental adaptation module fully exploits both the critical anomaly samples and the freshest normal samples to keep the classifier up to date. Extensive experiments on five real-world time-series datasets are conducted to assess FS-ADAPT, which outperforms the state-of-the-art FSL and UDA based time-series classification models, as well as their naive combinations.

Overview:

Load data, augment data, and construct 4-tuples (datasets.py)

Define networks including feature extractor, triplet network, and classifiers (networks.py)

Train model (main.py)

Environment:

Requires [mindspore](https://www.mindspore.cn/install) 


# MindSpore FS-ADAPT Implementation

This project implements the FS-ADAPT.

### Deprecation notice
This is an example experiment not the full one because file size exceeds capacity limit.
To run the code
```python
python main.py
```

Citation:

If you use this code in your research, please cite our paper "Few-Shot Time-Series Anomaly Detection with Unsupervised Domain Adaptation".

