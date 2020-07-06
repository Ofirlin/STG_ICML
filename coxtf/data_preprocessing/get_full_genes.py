import pickle
import random
import pandas as pd
import time

datapath = '../../data/coxtf/Metabric/metabric_exp.txt'

tic = time.time()

reader = pd.read_csv(datapath, sep='\t', chunksize=5000)

df = pd.concat((r for r in reader), ignore_index=False)

print(df.shape)

print('Time to concat: {}'.format(time.time() - tic))

tic = time.time()
df.to_csv('../../data/coxtf/Metabric/metabric_exp')
print('Time to save: {}'.format(time.time() - tic))

