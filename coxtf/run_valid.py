import utils
from nnsurv import L2DeepSurv
import json
import os
import numpy as np
import utils
import time

import argparse 
parser = argparse.ArgumentParser(description='Define parameters.')
parser.add_argument('--dataset', type=str)
parser.add_argument('--debug', action='store_true', help='debug mode')
args = parser.parse_args()

dataset_path = "../../data/stg/coxtf"
params_path = "./params/"
dataset_type = args.dataset 
data_dict = {'gaussian': 'gaussian_survival_data.h5', 'gbsg':'gbsg_cancer_train_test.h5',
        'linear':'linear_survival_data.h5', 'support':'support_train_test.h5',
        'treatment':'sim_treatment_dataset.h5', 'whas':'whas_train_test.h5',
        'metabric' : 'metabric_IHC4_clinical_train_test.h5',
        'oncotype' : 'oncotype_train_test.h5',
        'oncotyperand' : 'random_onco_train_valid_test.h5',
        'metabric_full': '0metabric_full_train_valid_test.h5'} 
raw_datafile = data_dict[dataset_type] #"support_train_test.h5"
run_name = 'run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
save_dir = "./results/" + run_name + '-' +dataset_type 
n_runs =1 

def main():
    with open(params_path + 'params_' + args.dataset + '.json') as f:
        params = json.load(f)
    print(params)

 

    CI_list = np.zeros(n_runs)
    loss_list = np.zeros(n_runs)
    tic = time.time()
    for i in range(n_runs):
        dataset_name = dataset_path+'/'+dataset_type+'/'+str(i)+raw_datafile
        print(dataset_name)
        datasets = utils.load_datasets(dataset_path+'/'+dataset_type+'/'+raw_datafile)
        train_data = datasets['train']
        norm_vals = {
                'mean' : datasets['train']['x'].mean(axis=0),
                'std'  : datasets['train']['x'].std(axis=0)
            }
        test_data = datasets['test']
        print(params['standardize'])
        if params['standardize'] == True:
            train_data = utils.standardize_dataset(datasets['train'], norm_vals['mean'], norm_vals['std'])
            valid_data = utils.standardize_dataset(datasets['valid'], norm_vals['mean'], norm_vals['std'])
            test_data = utils.standardize_dataset(datasets['test'], norm_vals['mean'], norm_vals['std'])

        print('dataset shape: \n')
        print('train x: {}, valid x: {}, test x: {}'.format(train_data['x'].shape, valid_data['x'].shape, test_data['x'].shape))
        print('train e: {}, valid e: {}, test e: {}'.format(train_data['e'].shape, valid_data['e'].shape, test_data['e'].shape))
        print('train t: {}, valid t: {}, test t: {}'.format(train_data['t'].shape, valid_data['t'].shape, test_data['t'].shape))
     
        train_X = train_data['x']
        train_y = {'e': train_data['e'], 't': train_data['t']}
        valid_X = valid_data['x']
        valid_y = {'e': valid_data['e'], 't': valid_data['t']}
        test_X = test_data['x']
        test_y = {'e': test_data['e'], 't': test_data['t']}
        input_nodes = train_X.shape[1]
        output_nodes = 1


        test_data = {}
        valid_data = {}
        '''
        test_data['X'], test_data['E'], \
            test_data['T'], test_data['failures'], \
            test_data['atrisk'], test_data['ties'] = utils.parse_data(test_X, test_y)
        '''
        test_data['X'], test_data['E'], \
                test_data['T'] = utils.prepare_data(test_X, test_y)
        test_data['ties']='noties'

        valid_data['X'], valid_data['E'], \
                valid_data['T'] = utils.prepare_data(valid_X, valid_y)
        valid_data['ties']='noties'

        test_label = {'t': test_data['T'], 'e': test_data['E']}
        valid_label = {'t': valid_data['T'], 'e': valid_data['E']}



        model = L2DeepSurv(train_X, train_y, valid_X, valid_y,
                           input_nodes, output_nodes, **params)

        output_dir = save_dir + '/run_' + str(i)
        if not os.path.exists(output_dir) and not args.debug:
            os.makedirs(output_dir)

        with open(os.path.join(output_dir + 'params.json'), 'w') as outfile:
            json.dump(params, outfile)

        # Plot curve of loss and CI on train data
        model.train(num_epoch=2000, iteration=100,
                    plot_loss=True, plot_CI=True, plot_gate=True, output_dir=output_dir)
        test_CI, test_loss = model.eval(test_data['X'], test_label)
        fin_val_CI, fin_val_loss = model.eval(valid_data['X'], valid_label)
        CI_list[i] = test_CI
        loss_list[i] = test_loss
    np.save(output_dir+'CI_list', CI_list)
    np.save(output_dir+'loss_list', loss_list)
    print("final valid CI: {}".format(fin_val_CI))
    print("final valid loss: {}".format(fin_val_loss))
    print("average test CI over {} runs:  {}".format(n_runs, CI_list.mean()))
    print("average test loss over {} runs:  {}".format(n_runs, loss_list.mean()))
    tac = time.time()
    print("Time: {}".format(tac - tic))

main()

