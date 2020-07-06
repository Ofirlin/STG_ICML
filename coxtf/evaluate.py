import tensorflow as tf
from nnsurv import L2DeepSurv
import numpy as np
import argparse
import json
import os
import utils
import time

parser = argparse.ArgumentParser(description='Define parameters.')
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

dataset_path = "../dataset"
params_path = "./params/"
dataset_type = args.dataset
data_dict = {'gaussian': 'gaussian_survival_data.h5', 'gbsg':'gbsg_cancer_train_test.h5',
        'linear':'linear_survival_data.h5', 'support':'support_train_test.h5',
        'treatment':'sim_treatment_dataset.h5', 'whas':'whas_train_test.h5',
        'metabric_full': '0metabric_full_train_valid_test.h5'}
raw_datafile = data_dict[dataset_type] #"support_train_test.h5"
save_dir = "./results/" + dataset_type


def main():
    # load dataset
    datasets = utils.load_datasets(dataset_path+'/'+dataset_type+'/'+raw_datafile)
    train_data = datasets['train']
    test_data = datasets['test']

    print('dataset shape: \n')
    print('train x: {}, test x: {}'.format(train_data['x'].shape, test_data['x'].shape))
    print('train e: {}, test e: {}'.format(train_data['e'].shape, test_data['e'].shape))
    print('train t: {}, test t: {}'.format(train_data['t'].shape, test_data['t'].shape))

    train_X = train_data['x']
    train_y = {'e': train_data['e'], 't': train_data['t']}
    test_X = test_data['x']
    test_y = {'e': test_data['e'], 't': test_data['t']}
    input_nodes = train_X.shape[1]
    output_nodes = 1

    with open(params_path + 'params_' + args.dataset + '.json') as f:
        params = json.load(f)

    test_data = {}
    test_data['X'], test_data['E'], \
        test_data['T'], test_data['failures'], \
        test_data['atrisk'], test_data['ties'] = utils.parse_data(test_X, test_y)
    test_label = {'t': test_data['T'], 'e': test_data['E']}

    model = L2DeepSurv(train_X, train_y, test_X, test_y,
		       input_nodes, output_nodes, **params)
    model_path = "./results/" + dataset_type + "/run_0/model-1"
    model.load(model_path)      
    test_CI, test_loss = model.eval(test_data['X'], test_label)
    print("loss: {}, CI: {}".format(test_loss, test_CI))

main()
