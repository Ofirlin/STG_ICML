print("__name__: {}".format(__name__))
print("__package__: {}".format(__package__))
print("__file__: {}".format(__file__))

from stg import STG

import unittest, sys
sys.path.insert(0, "/Users/yutaro/code/stg")

from examples.dataset import create_twomoon_dataset

from sklearn.model_selection import train_test_split


import numpy as np
import time

class Test(object):
    def __init__(self):
        n_size = 1000 #Number of samples
        p_size = 20   #Number of features
        X_data, y_data=create_twomoon_dataset(n_size,p_size)
        print(X_data.shape)
        print(y_data.shape)
        np.random.seed(123)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_data, y_data, train_size=0.3)
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train, self.y_train, train_size=0.8)

        self.params = {'hidden_layers_node': [60, 20, 2], 'param_search': False, 
                        'display_step': 1000, 'activation': 'tanh', 'lam': 0.02, 
                        'sigma': 0.5, 'feature_selection': True, 'learning_rate': 0.01, 
                        'output_node': 2}
        # Adjust params
        self.params['lam'] = 0.5
        self.params['learning_rate'] = 0.1
        self.params['input_node'] = self.X_train.shape[1]
        self.params['batch_size'] = self.X_train.shape[0]

    def test_torch(self):
        now = time.time()
        model = STG(task_type='classification',input_dim=self.X_train.shape[1], output_dim=2, hidden_dims=[60, 20], activation='tanh',
            optimizer='SGD', learning_rate=0.1, batch_size=self.X_train.shape[0], sigma=0.5, lam=0.5, random_state=1)
        model.fit(self.X_train, self.y_train, nr_epochs=5000, valid_X=self.X_valid, valid_y=self.y_valid, print_interval=1000)
        print("Passed time: {}".format(time.time() - now))


if __name__=='__main__':
    test = Test()
    test.test_torch()