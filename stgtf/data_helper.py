from sklearn.datasets import make_moons
from scipy.stats import norm
import numpy as np
import os
import pandas as pd


def create_sin_dataset(n,p):
    x1=5*(np.random.uniform(0,1,n)).reshape(-1,1)
    x2=5*(np.random.uniform(0,1,n)).reshape(-1,1)
    y=np.sin(x1)*np.cos(x2)**3
    relevant=np.hstack((x1,x2))
    noise_vector = norm.rvs(loc=0, scale=1, size=[n,p-2])
    data = np.concatenate([relevant, noise_vector], axis=1)
    
    return data, y
def create_twomoon_dataset(n, p):
    relevant, y = make_moons(n_samples=n, shuffle=True, noise=0.1, random_state=None)
    print(y.shape)
    noise_vector = norm.rvs(loc=0, scale=1, size=[n,p-2])
    data = np.concatenate([relevant, noise_vector], axis=1)
    print(data.shape)
    return data, y

def create_xor_dataset(n, p):
    var=1
    mean_z=np.zeros((2))
    sig= 1*np.random.multivariate_normal(mean_z, 1*np.eye(2), n)
    X_L=sig

    X_L[:,:]=(np.sign(X_L[:,:])+1)/2
    #X_L[np.where(X_L==-1),:]=0
    Y_L=X_L[:,0].astype(int)^X_L[:,1].astype(int)
    mean_z=np.zeros((p))
    noise = 1*np.random.multivariate_normal(mean_z, var*np.eye(p), n)
    noise_b=(np.sign(noise)+1)/2
    X_f=np.vstack((X_L.T,noise_b.T)).T
    X_f=X_f*2-1
    return X_f,Y_L

#def srff_syn_data_pure(exp_code):
def get_reg_data(exp_code):
    #base_path = './Pure_SE2_1000_001'
    base_path = './Pure_SE2_Dim2_1000_001'
    base_path = "/data/yutaro/ICML2019/SRFF/Experiments/Data/Pure_SE2_Dim2_1000_001"
    base_path = "/data/yutaro/ICML2019/SRFF/Experiments/Data/SE2_1000_001"
    base_path = os.path.join("/data/yutaro/ICML2019/SRFF/Experiments/Data/", exp_code)
    #base_path = './Pure_SE2_Dim2_50000_001'
    #base_path = './Pure_SE4_1000_001'
    #base_path = "/data/yutaro/ICML2019/SRFF/Experiments/Data/Pure_SE5_1000_001"
    train_x = pd.read_csv(os.path.join(base_path, 'trainX.csv'), header=None).values.astype(float)
    train_y = pd.read_csv(os.path.join(base_path, 'trainy.csv'), header=None).values.astype(float)
    valid_x = pd.read_csv(os.path.join(base_path, 'validX.csv'), header=None).values.astype(float)
    valid_y = pd.read_csv(os.path.join(base_path, 'validy.csv'), header=None).values.astype(float)
    test_x = pd.read_csv(os.path.join(base_path, 'testX.csv'), header=None).values.astype(float)
    test_y = pd.read_csv(os.path.join(base_path, 'testy.csv'), header=None).values.astype(float)
    return train_x, train_y, valid_x, valid_y, test_x, test_y

