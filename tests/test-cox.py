print("__name__: {}".format(__name__))
print("__package__: {}".format(__package__))
print("__file__: {}".format(__file__))
#from stgtf.utilsN import Model as Model_tf
#from stgtf.utilsN import DataSet, convertToOneHot, get_date_time

import unittest, sys
sys.path.insert(0, "/Users/yutaro/code/stg")
from stg import STG
import stg.utils as utils
from stg.losses import PartialLogLikelihood

from examples.dataset import create_twomoon_dataset

from sklearn.model_selection import train_test_split


import torch
import unittest, sys
import numpy as np
import time
import pandas as pd

import tensorflow as tf

class Test(object):
    def __init__(self):
        n_size = 1000 #Number of samples
        p_size = 20   #Number of features
        #X_data, y_data=create_twomoon_dataset(n_size,p_size)
        #print(X_data.shape)
        #print(y_data.shape)
        standardize = True
        np.random.seed(123)

        dataset_path = "../data/stg/coxtf"
        params_path = "./params/"
        dataset_type = 'oncotyperand' #'metabric_full' #args.dataset 
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


        #TODO: datast_name, dataset_path, dataset_type, raw_datafile?
        #i=0
        #dataset_name = dataset_path+'/'+dataset_type+'/'+str(i)+raw_datafile
        dataset_name = dataset_path+'/'+dataset_type+'/'+raw_datafile
        print(dataset_name)
        datasets = utils.load_datasets(dataset_path+'/'+dataset_type+'/'+raw_datafile)
        train_data = datasets['train']
        norm_vals = {
                'mean' : datasets['train']['x'].mean(axis=0),
                'std'  : datasets['train']['x'].std(axis=0)
            }
        test_data = datasets['test']
        if standardize == True:
            train_data = utils.standardize_dataset(datasets['train'], norm_vals['mean'], norm_vals['std'])
            valid_data = utils.standardize_dataset(datasets['valid'], norm_vals['mean'], norm_vals['std'])
            test_data = utils.standardize_dataset(datasets['test'], norm_vals['mean'], norm_vals['std'])

        train_X = train_data['x']
        train_y = {'e': train_data['e'], 't': train_data['t']}
        valid_X = valid_data['x']
        valid_y = {'e': valid_data['e'], 't': valid_data['t']}
        test_X = test_data['x']
        test_y = {'e': test_data['e'], 't': test_data['t']}
        input_nodes = train_X.shape[1]
        output_nodes = 1


        self.train_data={}
        self.train_data['X'], self.train_data['E'], \
                self.train_data['T'] = utils.prepare_data(train_X, train_y)
        self.train_data['ties']='noties'
        self.valid_data={}
        self.valid_data['X'], self.valid_data['E'], \
                self.valid_data['T'] = utils.prepare_data(valid_X, valid_y)
        self.valid_data['ties']='noties'
        self.test_data = {}
        self.test_data['X'], self.test_data['E'], \
                self.test_data['T'] = utils.prepare_data(test_X, test_y)
        self.test_data['ties']='noties'

    def test_tf_partial_likelihood(self):
        import tensorflow as tf
        ## Hyerparams
        input_node = self.train_data['X'].shape[1] 
        output_node = 1
        hidden_layers_node = [60, 20, 3]
        activation = 'selu'
        L1_reg=0.0002
        L2_reg=0.0003
        dropout_keep_prob=1.0
        feature_selection=False
        self.sigma=0.5
        def _to_tensor(x, dtype):
            """Convert the input `x` to a tensor of type `dtype`.
            # Arguments
                x: An object to be converted (numpy array, list, tensors).
                dtype: The destination type.
            # Returns
                A tensor.
            """
            return tf.convert_to_tensor(x, dtype=dtype)
        def hard_sigmoid(x):
            """Segment-wise linear approximation of sigmoid.
            Faster than sigmoid.
            Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
            In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.
            # Arguments
                x: A tensor or variable.
            # Returns
                A tensor.
            """
            x = x + 0.5
            zero = _to_tensor(0., x.dtype.base_dtype)
            one = _to_tensor(1., x.dtype.base_dtype)
            x = tf.clip_by_value(x, zero, one)
            return x
        def feature_selector(prev_x, train_gates):
            '''
            :param x: input
            :return:  gated input
            '''
            # gaussian reparametrization
            base_noise = tf.random_normal(shape=tf.shape(prev_x), mean=0., stddev=1.)
            z = tf.expand_dims(self.alpha, axis=0) + self.sigma * base_noise * train_gates
            stochastic_gate = hard_sigmoid(z)
            new_x = prev_x * stochastic_gate
            return new_x
        # New Graph
        G = tf.Graph()
        with G.as_default():
            # Data input
            X = tf.placeholder(tf.float32, [None, input_node], name = 'x-Input')
            y_ = tf.placeholder(tf.float32, [None, output_node], name = 'label-Input')
            train_gates = tf.placeholder(tf.float32, [1], name='train_gates')
            # hidden layers
            self.nnweights = [] # collect weights of network
            prev_node = input_node
            prev_x = X
            if feature_selection:
                with tf.variable_scope('gates', reuse=tf.AUTO_REUSE): 
                    self.alpha = tf.get_variable('alpha', [prev_node,], 
                                            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                    prev_x = feature_selector(prev_x, train_gates) 
                
            for i in range(len(hidden_layers_node)):
                layer_name = 'layer' + str(i+1)
                with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                    weights = tf.get_variable('weights', [prev_node, hidden_layers_node[i]], 
                                              initializer=tf.truncated_normal_initializer(stddev=0.1))
                    self.nnweights.append(weights)

                    biases = tf.get_variable('biases', [hidden_layers_node[i]],
                                             initializer=tf.constant_initializer(0.0))

                    layer_out = tf.nn.dropout(tf.matmul(prev_x, weights) + biases, dropout_keep_prob)
                    #layer_out = tf.matmul(prev_x, weights) + biases

                    if activation == 'relu':
                        layer_out = tf.nn.relu(layer_out)
                    elif activation == 'selu':
                        layer_out = tf.nn.selu(layer_out)
                    elif activation == 'sigmoid':
                        layer_out = tf.nn.sigmoid(layer_out)
                    elif activation == 'tanh':
                        layer_out = tf.nn.tanh(layer_out)
                    else:
                        raise NotImplementedError('activation not recognized')

                    prev_node = hidden_layers_node[i]
                    prev_x = layer_out
            # output layers 
            layer_name = 'layer_last'
            with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE):
                weights = tf.get_variable('weights', [prev_node, output_node], 
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
                self.nnweights.append(weights)

                biases = tf.get_variable('biases', [output_node],
                                         initializer=tf.constant_initializer(0.0))

                layer_out = tf.matmul(prev_x, weights) + biases
            # Output of Network
            y = layer_out
            # Global step
            with tf.variable_scope('training_step', reuse=tf.AUTO_REUSE):
                global_step = tf.get_variable("global_step", [], 
                                              dtype=tf.int32,
                                              initializer=tf.constant_initializer(0), 
                                              trainable=False)
            # Loss value
            ## L1 - L2 Regularization
            reg_item = tf.contrib.layers.l1_l2_regularizer(L1_reg,
                                                           L2_reg)
            reg_term = tf.contrib.layers.apply_regularization(reg_item, self.nnweights)
            if feature_selection:
                ## Regularization
                reg = 0.5 - 0.5*tf.erf((-1/(2) - self.alpha)/(self.sigma*np.sqrt(2)))
                reg_gates = tf.reduce_mean(reg) * self.lam
            ## Negative log likelihood
            loss_fun = self._negative_log_likelihood(y_, y)
            if feature_selection:
                loss = loss_fun + reg_term + reg_gates
            else:
                loss = loss_fun + reg_term
            params_dict = {}
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                N = self.train_data['E'].shape[0]
                tf_loss, tf_loss_fun, tf_logits, tf_fail_ind = sess.run([loss, loss_fun, y, y_],
                                   feed_dict = {X:  self.train_data['X'],
                                                y_: self.train_data['E'].reshape((N, 1)),
                                                train_gates: [1.0]})
                #for var in self.model_tf.trainable_variables:
                for var in tf.compat.v1.trainable_variables():
                    params_dict[var.name] = sess.run(var)
            for key, val in params_dict.items():
                print("{}, shape: {}".format(key, val.shape))
            #return params_dict, tf_out_np 
            print("loss: {}, logits: {}".format(tf_loss, tf_logits))
        return tf_loss, tf_loss_fun, tf_logits, tf_fail_ind, params_dict

    def _negative_log_likelihood(self, y_true, y_pred):
        """
        Callable loss function for DeepSurv network.
        the negative average log-likelihood of the prediction
        of this model under a given target distribution. 
        Parameters: 
            y_true: tensor, observations. 
            y_pred: tensor, output of network.
        Returns:
            loss value, means negative log-likelihood. 
        """
        logL = 0
        # pre-calculate cumsum
        cumsum_y_pred = tf.cumsum(y_pred)
        hazard_ratio = tf.exp(y_pred)
        cumsum_hazard_ratio = tf.cumsum(hazard_ratio)
        if self.train_data['ties'] == 'noties':
            log_risk = tf.log(cumsum_hazard_ratio)
            likelihood = y_pred - log_risk
            # dimension for E: np.array -> [None, 1]
            uncensored_likelihood = likelihood * y_true
            logL = -tf.reduce_sum(uncensored_likelihood)
        else:
            # Loop for death times
            for t in self.train_data['failures']:                                                                       
                tfail = self.train_data['failures'][t]
                trisk = self.train_data['atrisk'][t]
                d = len(tfail)
                dr = len(trisk)

                logL += -cumsum_y_pred[tfail[-1]] + (0 if tfail[0] == 0 else cumsum_y_pred[tfail[0]-1])

                if self.train_data['ties'] == 'breslow':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    logL += tf.log(s) * d
                elif self.train_data['ties'] == 'efron':
                    s = cumsum_hazard_ratio[trisk[-1]]
                    r = cumsum_hazard_ratio[tfail[-1]] - (0 if tfail[0] == 0 else cumsum_hazard_ratio[tfail[0]-1])
                    for j in range(d):
                        logL += tf.log(s - j * r / d)
                else:
                    raise NotImplementedError('tie breaking method not recognized')
        # negative average log-likelihood
        observations = tf.reduce_sum(y_true)
        return logL / observations

    def test_torch_partial_likelihood(self, state_dict):
        print(self.train_data['X'].shape)
        print(self.train_data['E'].shape)
        N = self.train_data['E'].shape[0]
        from stg.layers import MLPLayer
        model = MLPLayer(input_dim=self.train_data['X'].shape[1], output_dim=1, hidden_dims=[60,20,3],
                         batch_norm=None, dropout=None, activation='selu')
        self.torch_printParam(model)
        model.load_state_dict(state_dict)
        logits = model(torch.from_numpy(self.train_data['X']))
        loss = PartialLogLikelihood(logits, 
                    torch.from_numpy(self.train_data['E']).float().reshape(N,1), 'noties')    
        print(loss)
        #print(logits)
        return loss, logits, self.train_data['E']

    def test_compare_partial_likelihood(self):
        N = self.train_data['E'].shape[0]
        logits = np.random.randn(N,1)
        torch_logits = torch.from_numpy(logits).float()
        torch_loss = PartialLogLikelihood(torch_logits, torch.from_numpy(self.train_data['E']).float(), 'noties')
        print(torch_loss)
        tf_loss = self._negative_log_likelihood(tf.constant(self.train_data['E'], dtype=tf.float32), tf.constant(logits, dtype=tf.float32))
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            tf_loss_np = sess.run(tf_loss)
        print('tf: {}'.format(tf_loss_np))

    def torch_printParam(self, model_torch):
        for param_tensor in model_torch.state_dict():
            print(param_tensor, "\t", model_torch.state_dict()[param_tensor].size())

    def test_torch(self):
        args_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if args_cuda else "cpu")
        #torch.backends.cudnn.benchmark = True
        feature_selection = True #False 
        model = STG(task_type='cox',input_dim=self.train_data['X'].shape[1], output_dim=1, hidden_dims=[60, 20, 3], activation='selu',
            optimizer='Adam', learning_rate=0.0005, batch_size=self.train_data['X'].shape[0], feature_selection=feature_selection, 
            sigma=0.5, lam=0.004, random_state=1, device=device)
        now = time.time()
        model.fit(self.train_data['X'], {'E': self.train_data['E'], 'T':self.train_data['T']}, nr_epochs=600, 
                valid_X=self.valid_data['X'], valid_y={'E':self.valid_data['E'], 'T':self.valid_data['T']}, print_interval=100)
        print("Passed time: {}".format(time.time() - now))
        if feature_selection:
            #print(model.get_gates(mode='prob'))
            pass
        
        model.evaluate(self.test_data['X'], {'E': self.test_data['E'], 'T':self.test_data['T']})

        model.save_checkpoint(filename='tmp.pth')
    
    def test_load(self):
        filename='tmp.pth'
        args_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if args_cuda else "cpu")
        #torch.backends.cudnn.benchmark = True
        feature_selection = True #False 
        model = STG(task_type='cox',input_dim=self.train_data['X'].shape[1], output_dim=1, hidden_dims=[60, 20, 3], activation='selu',
            optimizer='Adam', learning_rate=0.0005, batch_size=self.train_data['X'].shape[0], feature_selection=feature_selection, 
            sigma=0.5, lam=0.004, random_state=1, device=device)
        
        model.load_checkpoint(filename)
        model.evaluate(self.test_data['X'], {'E': self.test_data['E'], 'T':self.test_data['T']})

    '''
    def test_torch(self):
        now = time.time()
        model = STG(task_type='cox',input_dim=self.train_data['X'].shape[1], output_dim=1, hidden_dims=[60, 20], activation='tanh',
            optimizer='SGD', learning_rate=0.1, batch_size=self.X_train.shape[0], sigma=0.5, lam=0.5, random_state=1)
        #model.fit(self.X_train, self.y_train, nr_epochs=5000, valid_X=self.X_valid, valid_y=self.y_valid, print_interval=1000)
        print("Passed time: {}".format(time.time() - now))
    '''

    def print_format(self, v, names, max_rows):
        assert len(v) == len(names)
        if isinstance(v, list):
            dict = {}
            for arr, name in zip(v, names):
                dict.update({name: arr})
            #x = pd.DataFrame.from_records([dict])
            x = pd.DataFrame(dict)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
            print(x)
            #print(x.head(max_rows))

    def params_convert(self, tf_params_dict):
        tf_2_torch = {}
        tf_2_torch.update({
            "layer1/weights:0":"mlp.0.0.weight", #shape: (221, 60) torch.Size([60, 221])
            "layer1/biases:0":"mlp.0.0.bias", #shape: (60,) torch.Size([60])
            "layer2/weights:0":"mlp.1.0.weight", #shape: (60, 20) torch.Size([20, 60])
            "layer2/biases:0":"mlp.1.0.bias", #shape: (20,) torch.Size([20])
            "layer3/weights:0":"mlp.2.0.weight", #shape: (20, 3) torch.Size([3, 20])
            "layer3/biases:0":"mlp.2.0.bias", #shape: (3,) torch.Size([3])
            "layer_last/weights:0":"mlp.3.weight", #shape: (3, 1) torch.Size([1, 3])
            "layer_last/biases:0":"mlp.3.bias", #shape: (1,) torch.Size([1])
        })
        #tf_2_torch.update({            
        #    "gates/alpha:0": "",  #shape: (221,)
        #    })
        state_dict = {}
        for key, val in tf_params_dict.items():
            if tf_2_torch[key].split('.')[-1] == 'weight':
                state_dict[tf_2_torch[key]] = torch.from_numpy(val).permute(1,0)
            elif tf_2_torch[key].split('.')[-1] == 'bias':
                state_dict[tf_2_torch[key]] = torch.from_numpy(val) #.permute(2,0,1)
            else:
                state_dict[tf_2_torch[key]] = torch.from_numpy(val)
        return state_dict


if __name__=='__main__':
    test = Test()
    test.test_torch()
    '''
    tf_loss, tf_loss_fun, tf_logits, tf_fail_ind, tf_params_dict = test.test_tf_partial_likelihood()
    state_dict = test.params_convert(tf_params_dict)
    torch_loss, torch_logits, torch_fail_ind = test.test_torch_partial_likelihood(state_dict)
    print("TF_loss_fun: {}, torch_loss: {}".format(tf_loss_fun, torch_loss))
    print("TF_loss: {}".format(tf_loss))
    test.print_format([tf_logits.reshape(-1), torch_logits.detach().numpy().reshape(-1)],
                    ['tf_logits', 'torch_logits'],
                    200)
    test.print_format([tf_fail_ind.reshape(-1), torch_fail_ind.reshape(-1)],
                    ['tf_fail_ind', 'torch_fail_ind'],
                    200)
    '''
    #test.test_compare_partial_likelihood()

    #test.test_load()