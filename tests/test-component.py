print(__name__)
print(__package__)
print(__file__)
from stgtf.utilsN import Model as Model_tf
from stgtf.utilsN import DataSet, convertToOneHot, get_date_time

#from stg import STG
import numpy as np
import time
import unittest, sys
sys.path.insert(0, "/Users/yutaro/code/stg")
from models import STGClassificationModel

from examples.dataset import create_twomoon_dataset

from sklearn.model_selection import train_test_split



class Test(unittest.TestCase):
    def setUp(self):
        self.model_torch = STGClassificationModel(input_dim, output_dim, hidden_dims, activation=activation, sigma=sigma, lam=lam)
        pass

    def tearDown(self):
        pass

    def test_logits(self):
        pass

if __name__=='__main__':
    unittest.main()