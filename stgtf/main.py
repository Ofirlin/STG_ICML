# main driver
from data_helper import create_twomoon_dataset, create_xor_dataset, get_reg_data
import argparse
import json
from sklearn.model_selection import train_test_split
from utils import DataSet, convertToOneHot, get_date_time
from model import Model
import time
import os
import sys

parser = argparse.ArgumentParser(description='Define parameters for experiments.')
parser.add_argument('--dataset_type', type=str)
parser.add_argument('--task_type', type=str) # regression or classification
parser.add_argument('--n_trials', type=int) # how many times we repeat for conf interval
parser.add_argument('--cuda_id', type=int)
parser.add_argument('--num_epoch',type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)

params_path = "./params/"

def main(model_dir, exp_code):
    with open(params_path + 'params_' + args.dataset_type + '.json') as f:
        params = json.load(f)
    print(params)

    if args.task_type == 'classification':
        if args.dataset_type == 'twomoons':
            X_data, y_data = create_twomoon_dataset(params.n_size, params.p_size)
        elif args.dataset_type == 'xor':
            X_data, y_data = create_xor_dataset(params.n_size, params.p_size)
        else:
            raise NotImplementedError('dataset name not recognized')

        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size=0.3)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.8)
        print("X_train: {}".format(X_train.shape))
        print("X_valid: {}".format(X_valid.shape))
        print("X_test: {}".format(X_test.shape))
        print("y_train: {}".format(y_train.shape))
        print("y_valid: {}".format(y_valid.shape))
        print("y_test: {}".format(y_test.shape))

        y_train=convertToOneHot(y_train.astype(int))
        y_valid=convertToOneHot(y_valid.astype(int))
        y_test=convertToOneHot(y_test.astype(int))
    elif args.task_type == 'regression':
        # NOTE: get_reg_data(args.dataset_type) used to be srff_syn_data_pure(exp_code)
        X_train, y_train, X_valid, y_valid, X_test, y_test = get_reg_data(exp_code)

    params['input_node'] = X_train.shape[1]
    params['batch_size']= X_train.shape[0]

    dataset = DataSet(**{'_data':X_train, '_labels':y_train,
                    '_valid_data':X_valid, '_valid_labels':y_valid,
                    '_test_data':X_test, '_test_labels':y_test})
    train_acc_mat, train_loss_mat = [], []
    test_acc_list, test_loss_list = [], []
    with open(model_dir + '/params.json', 'w') as f:
        json.dump(params, f)
    for trial_id in range(args.n_trials):
        model = Model(**params)
        tic = time.time()
        train_acces, train_losses, val_acces, val_losses = model.train(params['param_search'], dataset, model_dir, num_epoch=args.num_epoch)
        model.save(step=trial_id, model_dir=model_dir)
        test_acc, test_loss = model.evaluate(X_test, y_test)
        tac = time.time()
        print("Time: {}".format(tac - tic))
        print(model.get_raw_alpha())
        prob_a = model.get_prob_alpha()
        print(prob_a)
        train_acc_mat.append(train_acces)
        train_loss_mat.append(train_losses)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
    return train_acc_mat, train_loss_mat, test_acc_list, test_loss_list

if __name__=='__main__':
    time_stamp = get_date_time()
    model_dir = './results/' + args.task_type +'/' + args.dataset_type +'/'+time_stamp
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if args.task_type == 'regression':
        assert args.n_trials == 1,"n_trials should be 1 for regression. Instead we loop over the datasets."
        test_loss_list = []
        for exp_id in range(1, 31):
            exp_id = str(exp_id).zfill(3)
            exp_code = args.dataset_type + exp_id
            train_acc_mat, train_loss_mat, test_acc, test_loss = main(model_dir, exp_code)
            test_loss_list.append(test_loss)
        print(np.array(test_loss_list).mean(), np.array(test_loss_list).std())
        with open(os.path.join(model_dir,"test_loss_list.pkl"), "wb") as f:
            pickle.dump(test_loss_list, f)
    elif args.task_type == 'classification':
        exp_code = 0
        train_acc_mat, train_loss_mat, test_acc_list, test_loss_list = main(model_dir, exp_code)
    else:
        raise NotImplementedError('task_type name not recognized')

