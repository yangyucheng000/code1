"""
Datasets

Load the desired datasets, augment data, and construct 4-tuples
"""
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
import multiprocessing
from collections import Counter
import mindspore.dataset as ds

MAX_DATA_SIZE = 220
SAVE_FILE = False
DOMAIN_DICT = {"Artificial": 0, "AWS": 1, "Twitter": 2, "Yahoo": 3}
temp_x = []
temp_y = []
temp_z = []

class Dataset:
    """Base class for datasets"""
    already_normalized = False

    def __init__(self, num_attrs, max_window_size, folder_path, test_percent=0.1):

        # Set parameters
        self.num_attrs = num_attrs # number of attributes
        self.max_window_size = max_window_size
        self.test_percent = test_percent
        self.folder_path = folder_path

    def load_data(self, folder_path):
        """ Obtain all attributes for one dataset """
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

        x_matrix_list = []  
        y_matrix_list = []  

        max_x_columns = 0  
        max_y_columns = 0

        for csv_file in csv_files:
            file_path = os.path.join(folder_path, csv_file)
            data = np.genfromtxt(file_path, delimiter=',', skip_header=1) 


            #Regularize each attribute
            scaler = MinMaxScaler()
            normalized_attribute = scaler.fit_transform(data[:, 1].reshape(-1, 1))
            data[:, 1] = normalized_attribute.flatten()  
            x_matrix_list.append(data[:, 1])
            y_matrix_list.append(data[:, 2])
            if max_x_columns < len(data[:, 1]):
                max_x_columns = len(data[:, 1])
            if max_y_columns < len(data[:, 2]):
                max_y_columns = len(data[:, 2])


        for i in range(len(x_matrix_list)):
            temp = max_x_columns - len(x_matrix_list[i])
            x_matrix_list[i]=np.pad(x_matrix_list[i], [(0, temp)],mode="constant", constant_values=0)
            y_matrix_list[i]=np.pad(y_matrix_list[i], [(0, temp)],mode="constant", constant_values=0)

        x_matrix = np.vstack(x_matrix_list)
        y_vector = []
        for col in range(max_y_columns):
            if all(row[col] == 0 for row in y_matrix_list):
                y_vector.append(0)
            else:
                y_vector.append(1)

        counts = Counter(y_vector)
        print(self.folder_path, x_matrix.shape, len(y_vector))
        print(self.folder_path,"number of the positive instances and negative instances in the original dataset：",counts[1],counts[0])
        return x_matrix, y_vector


    def augmentation(self, x, y, max_window_size):
        """ Augment time-series data """
        augments_x_normal = []
        augments_x_anomaly = []
        augments_y_normal = []
        augments_y_anomaly = []

        
        current_sublist = []
        for index, val in enumerate(y):
            if val == 0:
                current_sublist.append(index)
            else:
                if current_sublist:
                    if max(current_sublist) - min(current_sublist) <= max_window_size:
                        for window_size_normal1 in range(max(current_sublist) - min(current_sublist)):
                            augments_y_normal.append(0)
                            augments_x_normal.append(x[:, min(current_sublist):min(current_sublist)+window_size_normal1+1])
                    else:
                        for i in range(min(current_sublist),max(current_sublist),1):
                            if i + max_window_size <= max(current_sublist):
                                for window_size_normal2 in range(max_window_size):
                                    augments_y_normal.append(0)
                                    augments_x_normal.append(x[:, min(current_sublist):min(current_sublist) + window_size_normal2 + 1])
                                i = i + 1
                

        
        for i in range(len(y)-1,0,-1):
            if y[i] == 1:
                for window_size_anomaly1 in range(max_window_size):
                    augments_x_anomaly.append(x[:, i - window_size_anomaly1 - 1:i])
                    augments_y_anomaly.append(1)
                i = i - 1

        augments_x = augments_x_normal + augments_x_anomaly
        augments_y = augments_y_normal + augments_y_anomaly

        
        print(self.folder_path, "number of the positive instances and negative instances after augmentation：",len(augments_x_normal), len(augments_x_anomaly))

        return augments_x, augments_y

    def train_test_split(self, x, y, random_state=30):
        """ Split x and y data into train/test sets """

        
        x_train, x_test, y_train, y_test = \
            train_test_split(x, y, test_size=self.test_percent, stratify=y, random_state=random_state)
        
        return x_train, y_train, x_test, y_test


def organization(source_x_train, source_y_train, source_z_train, target_x_train, target_y_train, target_z_train, num_samples):
    """ Construct 4-tuples """
    train_tuples_x = []
    train_tuples_y = []
    train_tuples_z = []
    typical_normal = []
    typical_anomaly = []
    datasets_list = ["Yahoo","Artificial","AWS","Twitter"]
    source_domain_list = ["Yahoo", "Artificial", "AWS"]
    target_domain_list = ["Twitter"]
    global temp_x, temp_y, temp_z

    temp_x = source_x_train + target_x_train
    temp_y = source_y_train + target_y_train
    temp_z = source_z_train + target_z_train

    normal_indices = [i for i, y in enumerate(temp_y) if y == 0]
    typical_normal_index = random.sample(normal_indices, num_samples)
    for val in typical_normal_index:
        typical_normal.append(temp_x[val])
    anomaly_indices = [i for i, y in enumerate(temp_y) if y == 1]
    typical_anomaly_index = random.sample(anomaly_indices, num_samples)
    for val in typical_anomaly_index:
        typical_anomaly.append(temp_x[val])

    max_datasize = MAX_DATA_SIZE

    for i in range(min(len(source_x_train), max_datasize)):
        if source_y_train[i] == 1 and source_x_train[i].size > 0:
            sample_3 = source_x_train[i]
            source_domain = source_z_train[i]
            for j in range(min(len(source_x_train), max_datasize)):
                if source_y_train[j] == 1 and j != i and source_x_train[j].size > 0:
                    sample_2 = source_x_train[j]
                    for k in range(min(len(temp_x), max_datasize)):
                        if temp_z[k] != source_domain and temp_x[k].size > 0:
                            sample_1 = temp_x[k]
                            y_1 = temp_y[k]
                            z_1 = temp_z[k]
                            for l in range(min(len(source_x_train), max_datasize)):
                                if source_y_train[l] == 0 and source_x_train[l].size > 0:
                                    train_tuples_x.append((sample_1, sample_2, sample_3, source_x_train[l]))
                                    train_tuples_y.append((y_1, 1, 1, 0))
                                    train_tuples_z.append((DOMAIN_DICT[z_1], DOMAIN_DICT[source_domain], DOMAIN_DICT[source_domain], DOMAIN_DICT[source_z_train[l]]))
                                

    return train_tuples_x, train_tuples_y, train_tuples_z, typical_normal, typical_anomaly                              

def source_target(source_list, target,x_train_ya, x_train_ar, x_train_aw, x_train_tw, y_train_ya, y_train_ar, y_train_aw, y_train_tw, x_test_ya, x_test_ar, x_test_aw, x_test_tw, y_test_ya, y_test_ar, y_test_aw, y_test_tw,num_samples):
    """ Build source and target datasets """

    #print("source_target_start")
    target_x_train = []
    target_y_train = []
    target_z_train = []
    target_x_test = []
    target_y_test = []
    target_z_test = []
    if target == "Yahoo":
        target_x_train += x_train_ya
        target_y_train += y_train_ya
        target_z_train += ["Yahoo"] * len(y_train_ya)
        target_x_test += x_test_ya
        target_y_test += y_test_ya
        target_z_test += ["Yahoo"] * len(y_test_ya)
    elif target == "Artificial":
        target_x_train += x_train_ar
        target_y_train += y_train_ar
        target_z_train += ["Artificial"] * len(y_train_ar)
        target_x_test += x_test_ar
        target_y_test += y_test_ar
        target_z_test += ["Artificial"] * len(y_test_ar)
    elif target == "AWS":
        target_x_train += x_train_aw
        target_y_train += y_train_aw
        target_z_train += ["AWS"] * len(y_train_aw)
        target_x_test += x_test_aw
        target_y_test += y_test_aw
        target_z_test += ["AWS"] * len(y_test_aw)
    else:
        target_x_train += x_train_tw
        target_y_train += y_train_tw
        target_z_train += ["Twitter"] * len(y_train_tw)
        target_x_test += x_test_tw
        target_y_test += y_test_tw
        target_z_test += ["Twitter"] * len(y_test_tw)

    source_x_train = []
    source_y_train = []
    source_z_train = []
    source_x_test = []
    source_y_test = []
    source_z_test = []

    for name in source_list:
        if name == "Yahoo":
            source_x_train += x_train_ya
            source_y_train += y_train_ya
            source_z_train += ["Yahoo"]*len(y_train_ya)
            source_x_test += x_test_ya
            source_y_test += y_test_ya
            source_z_test += ["Yahoo"] * len(y_test_ya)
        if name == "Artificial":
            source_x_train += x_train_ar
            source_y_train += y_train_ar
            source_z_train +=["Artificial"]*len(y_train_ar)
            source_x_test += x_test_ar
            source_y_test += y_test_ar
            source_z_test +=["Artificial"]*len(y_test_ar)
        if name == "AWS":
            source_x_train += x_train_aw
            source_y_train += y_train_aw
            source_z_train +=["AWS"]*len(y_train_aw)
            source_x_test += x_test_aw
            source_y_test += y_test_aw
            source_z_test +=["AWS"]*len(y_test_aw)
        else:
            source_x_train += x_train_tw
            source_y_train += y_train_tw
            source_z_train +=["Twitter"]*len(y_train_tw)
            source_x_test += x_test_tw
            source_y_test += y_test_tw
            source_z_test +=["Twitter"]*len(y_test_tw)


    train_tuples_x, train_tuples_y, train_tuples_z, typical_normal, typical_anomaly = organization(source_x_train, source_y_train, source_z_train, target_x_train, target_y_train, target_z_train, num_samples)
    
    #remove zero in target sets:
    for i, x in enumerate(target_x_test):
        if x.size <= 0:
            del target_x_test[i]
            del target_y_test[i]
            del target_z_test[i]

    typical_normal = [x for x in typical_normal if x.size > 0]
    typical_anomaly = [x for x in typical_anomaly if x.size > 0]

    if SAVE_FILE:
        print("Writing data into files")
        with open('train_tuples_x.txt', "w") as file:
            for item in train_tuples_x:
                file.write(str(item) + "\n")
        with open('train_tuples_y.txt', "w") as file1:
            for item in train_tuples_y:
                file1.write(str(item) + "\n")
        with open('train_tuples_z.txt', "w") as file2:
            for item in train_tuples_z:
                file2.write(str(item) + "\n")
        with open('typical_normal.txt', "w") as file3:
            for item in typical_normal:
                file3.write(str(item) + "\n")
        with open('typical_anomaly.txt', "w") as file4:
            for item in typical_anomaly:
                file4.write(str(item) + "\n")
        print("Files writing done")
    return train_tuples_x, train_tuples_y, train_tuples_z, typical_normal, typical_anomaly, target_x_test, target_y_test, target_z_test


def load_and_augment_data(args):
    base_class, folder_path, max_window_size = args
    x_matrix, y_vector = base_class.load_data(folder_path)
    augments_x, augments_y = base_class.augmentation(x_matrix, y_vector, max_window_size)
    x_train, y_train, x_test, y_test = base_class.train_test_split(augments_x, augments_y)
    return x_train, y_train, x_test, y_test

class MsTrainDataset:
    def __init__(self, tuples, class_label, domain_label):
        self._tuples = tuples
        self._domain_lable = domain_label
        self._class_label = class_label
    
    def __getitem__(self, index):
        return self._tuples[index][0], self._tuples[index][1], self._tuples[index][2], self._tuples[index][3], self._domain_lable[index], self._class_label[index]

    def __len__(self):
        return len(self._tuples)

class MsTestDataset:
    def __init__(self, tuples, class_label, domain_label):
        self._tuples = tuples
        self._domain_lable = domain_label
        self._class_label = class_label
    
    def __getitem__(self, index):
        return self._tuples[index], self._domain_lable[index], self._class_label[index]

    def __len__(self):
        return len(self._tuples)

class MsOnlineTestDataset:
    def __init__(self, test_sample, class_label, domain_label):
        self._test_sample = test_sample
        self._domain_lable = domain_label
        self._class_label = class_label
    
    def __getitem__(self, index):
        return self._test_sample[index], self._domain_lable, self._class_label

    def __len__(self):
        return len(self._test_sample)

def get_input_shape():
    return 1

def get_random_positive_sample():
    normal_indices = [i for i, y in enumerate(temp_y) if y == 0]
    random_normal_index = random.sample(normal_indices, 1)
    return temp_x[random_normal_index[0]]

def get_random_negative_sample():
    while 1:
        anomaly_indices = [i for i, y in enumerate(temp_y) if y == 1]
        random_anomaly_index = random.sample(anomaly_indices, 1)
        if temp_x[random_anomaly_index[0]].size > 0:
            return temp_x[random_anomaly_index[0]]
    return 

def get_data():
    YahooBase = Dataset(num_attrs=1, max_window_size=20, folder_path="datasets/Yahoo")
    ArtificialBase = Dataset(num_attrs=1, max_window_size=20, folder_path="datasets/Artificial")
    AWSBase = Dataset(num_attrs=1, max_window_size=20, folder_path="datasets/AWS")
    TwitterBase = Dataset(num_attrs=1, max_window_size=20, folder_path="datasets/Twitter")

    pool = multiprocessing.Pool(processes=4)

    
    datasets = [
        (YahooBase, YahooBase.folder_path, YahooBase.max_window_size),
        (ArtificialBase, ArtificialBase.folder_path, ArtificialBase.max_window_size),
        (AWSBase, AWSBase.folder_path, AWSBase.max_window_size),
        (TwitterBase, TwitterBase.folder_path, TwitterBase.max_window_size),
    ]

    
    results = pool.map(load_and_augment_data, datasets)

    
    pool.close()
    pool.join()

    train_tuples_x, train_tuples_y, train_tuples_z, typical_normal, typical_anomaly, target_x_test, target_y_test, target_z_test = \
        source_target(["Yahoo","Artificial","AWS"], "Twitter", results[0][0], results[1][0], results[2][0], results[3][0], results[0][1], results[1][1], results[2][1], results[3][1], results[0][2], results[1][2], results[2][2], results[2][2], results[0][3], results[1][3], results[2][3], results[2][3],20)

    #Process tuple list into mindspore data pipeline
    loader = MsTrainDataset(train_tuples_x, train_tuples_y, train_tuples_z)
    train_dataset = ds.GeneratorDataset(source=loader, column_names=["x0", "x1", "x2", "x3", "class", "domain"])

    loader = MsTestDataset(target_x_test, target_y_test, target_z_test)
    test_dataset = ds.GeneratorDataset(source=loader, column_names=["x0", "class", "domain"])

    loader = MsOnlineTestDataset(typical_normal, 0, DOMAIN_DICT["Twitter"])
    online_test_dataset_normal = ds.GeneratorDataset(source=loader, column_names=["x0", "class", "domain"])

    loader = MsOnlineTestDataset(typical_anomaly, 1, DOMAIN_DICT["Twitter"])
    online_test_dataset_anomaly = ds.GeneratorDataset(source=loader, column_names=["x0", "class", "domain"])


    return train_dataset, test_dataset, online_test_dataset_normal, online_test_dataset_anomaly


if __name__ == "__main__":
    train_dataset, test_dataset, online_test_dataset_normal, online_test_dataset_anomaly = get_data()
    test_dataset = test_dataset.batch(batch_size=1)
