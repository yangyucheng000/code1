# FETCH-MindSpore

FETCH after migration from PyTorch to MindSpore.

Migration environment:
Ubuntu 20.04,
Miniconda conda3,
Python 3.8,
Cuda 11.8,
MindSpore 2.2.10.

To be tested...

|      数据集       | FETCH  | py-score | py-score(seed1) | mi-score(seed1) |
| :---------------: | :----: | :------: | :-------------: | :-------------: |
|      Airfoil      | 0.6463 | 0.650089 |   0.650594682   |   0.656503074   |
|    BikeShareDC    | 0.9997 | 0.999669 |   0.999681773   |   0.999692664   |
|   HousingBoston   | 0.5224 | 0.530273 |   0.515825871   |   0.540572457   |
|  winequality-red  | 0.6042 | 0.605376 |   0.609122257   |   0.604130094   |
| winequality-white | 0.5235 | 0.521647 |   0.518790728   |   0.524516062   |
|        ...        |  ...   |   ...    |       ...       |       ...       |


