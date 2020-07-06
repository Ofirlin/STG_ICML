import numpy as np
import random
import pandas as pd
from sklearn.model_selection import ShuffleSplit, train_test_split
from collections import defaultdict
import h5py
import copy

from dataset import SimulatedData

def prepare_data(x, label):
    if isinstance(label, dict):
       e, t = label['e'], label['t']

    # Sort Training Data for Accurate Likelihood
    sort_idx = np.argsort(t)[::-1]
    x = x[sort_idx]
    e = e[sort_idx]
    t = t[sort_idx]

    #return x, {'e': e, 't': t} this is for parse_data(x, label); see the third line in the parse_data function. 
    return x,  e,  t

def parse_data(x, label):
    # sort data by t
    x, label = prepare_data(x, label)
    e, t = label['e'], label['t']

    failures = {}
    atrisk = {}
    n, cnt = 0, 0

    for i in range(len(e)):
        if e[i]:
            if t[i] not in failures:
                failures[t[i]] = [i]
                n += 1
            else:
                # ties occured
                cnt += 1
                failures[t[i]].append(i)

            if t[i] not in atrisk:
                atrisk[t[i]] = []
                for j in range(0, i+1):
                    atrisk[t[i]].append(j)
            else:
                atrisk[t[i]].append(i)
    # when ties occured frequently
    if cnt >= n / 2:
        ties = 'efron'
        ties = 'noties'
    elif cnt > 0:
        ties = 'breslow'
        ties = 'noties'
    else:
        ties = 'noties'

    return x, e, t, failures, atrisk, ties

'''load_datasets function is from utils_jared.py'''
def load_datasets(dataset_file):
    datasets = defaultdict(dict)
    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    return datasets

'''standardize_dataset function is from utils_jared.py'''
def standardize_dataset(dataset, offset, scale):
    norm_ds = copy.deepcopy(dataset)
    norm_ds['x'] = (norm_ds['x'] - offset) / scale
    return norm_ds

def loadSimulatedData(hr_ratio=2000, n=2000, m=10, num_var=2, seed=1):
    data_config = SimulatedData(hr_ratio, num_var = num_var, num_features = m)
    data = data_config.generate_data(n, seed=seed)
    data_X = data['x']
    data_y = {'e': data['e'], 't': data['t']}
    return data_X, data_y

def loadData(filename = "data//surv_aly_idfs.csv", 
             tgt={'e': 'idfs_bin', 't': 'idfs_month'}, 
             split=1.0,
             Normalize=True,
             seed=40):
    data_all = pd.read_csv(filename)

    ID = 'patient_id'
    target = list(tgt.values())
    L = target + [ID]
    x_cols = [x for x in data_all.columns if x not in L]

    X = data_all[x_cols]
    y = data_all[target]
    # Normalized data
    if Normalize:
        for col in X.columns:
            X.loc[:, col] = (X.loc[:, col] - X.loc[:, col].mean()) / (X.loc[:, col].max() - X.loc[:, col].min())
    # Split data
    if split == 1.0:
        train_X, train_y = X, y
    else:
        sss = ShuffleSplit(n_splits = 1, test_size = 1-split, random_state = seed)
        for train_index, test_index in sss.split(X, y):
            train_X, test_X = X.loc[train_index, :], X.loc[test_index, :]
            train_y, test_y = y.loc[train_index, :], y.loc[test_index, :]
    # print information about train data
    print("Number of rows: ", len(train_X))
    print("X cols: ", len(train_X.columns))
    print("Y cols: ", len(train_y.columns))
    print("X.column name:", train_X.columns)
    print("Y.column name:", train_y.columns)
    # Transform type of data to np.array
    train_X = train_X.values
    train_y = {'e': train_y[tgt['e']].values,
               't': train_y[tgt['t']].values}
    if split == 1.0:
        return train_X, train_y
    else:
        test_X = test_X.values
        test_y = {'e': test_y[tgt['e']].values,
                  't': test_y[tgt['t']].values}
        return train_X, train_y, test_X, test_y

def loadRawData(filename = "data//train_idfs.csv", 
                discount=None,
                seed=1):
    # Get raw data(no header, no split, has been pre-processed)
    data_all = pd.read_csv(filename, header=None)
    num_features = len(data_all.columns)
    X = data_all.loc[:, 0:(num_features-3)]
    y = data_all.loc[:, (num_features-2):]
    # split data
    if discount is None or discount == 1.0:
        train_X, train_y = X, y
    else:
        sss = ShuffleSplit(n_splits=1, test_size=1-discount, random_state=seed)
        for train_index, test_index in sss.split(X, y):
            train_X, test_X = X.loc[train_index, :], X.loc[test_index, :]
            train_y, test_y = y.loc[train_index, :], y.loc[test_index, :]
    # print information about train data
    print("Shape of train_X: ", len(train_X.index), len(train_X.columns))
    print("Shape of train_y: ", len(train_y.index), len(train_y.columns))
    # Transform type of data to np.array
    train_X = train_X.values
    train_y = {'e': train_y.iloc[:, 0].values,
               't': train_y.iloc[:, 1].values}
    if discount is None or discount == 1.0:
        return train_X, train_y
    else:
        # print information about test data
        print("Shape of test_X: ", len(test_X.index), len(test_X.columns))
        print("Shape of test_y: ", len(test_y.index), len(test_y.columns))
        test_X = test_X.values
        test_y = {'e': test_y.iloc[:, 0].values,
                  't': test_y.iloc[:, 1].values}
        return train_X, train_y, test_X, test_y

def readData(file, name, discount, seed=1):
    random.seed(seed)
    # Read data
    data0 = pd.read_csv(file)
    # One-Hot encode
    data = pd.get_dummies(data0, prefix=None, prefix_sep="_", dummy_na=False, columns=name, drop_first=False)
    # Reorder columns 'idfs_month' and 'idfs_bin' to the last
    a=data.pop('idfs_month')
    data.insert(len(list(data)),'idfs_month',a)
    b=data.pop('idfs_bin')
    data.insert(len(list(data)),'idfs_bin',b)

    #data0=pd.Series.map()
    #data1=pd.get_dummies()
    names = np.array(list(data))

    dataMatrix = np.array(data)
    rowNum, colNum = dataMatrix.shape[0], dataMatrix.shape[1]
    sampleData = []
    sampleClass = []
    for i in range(0, rowNum):
        tempList = list(dataMatrix[i,:])
        sampleClass.append(tempList[colNum-1])
        sampleData.append(tempList[0:colNum-1])
    sampleM = np.array(sampleData)
    # for continuous variable
    sampleM[:, :3] -= np.mean(sampleM[:, :3], axis=0)

    # sampleM[:,:3] =(sampleM[:,:3] - np.mean(sampleM[:,:3], axis=0))/np.std(sampleM[:,:3], axis=0)
    # print(sampleM[:,:3])

    # sampleM[:,:3] = preprocessing.scale(sampleM[:,:3], axis=0)

    classM = np.array(sampleClass)

    X_train, X_test, y_train, y_test = train_test_split(sampleM, classM, train_size=discount,random_state=1,stratify=classM)

    train_data = {
        'x': X_train[:,:-1],
        't': np.around(X_train[:,-1], decimals=2),
        'e': y_train
    }

    test_data = {
        'x': X_test[:,:-1],
        't': np.around(X_test[:,-1], decimals=2),
        'e': y_test
    }

    return (train_data, test_data, names)
