# I just need to get a list of 'ILMN_*'s
# Generate a set of random ids
# Get the corresponding 'ILMN_*' ids.
import pickle
import random
import pandas as pd
import time

with open('ILMN_list.pkl', 'rb') as f:
    ILMN_list = pickle.load(f)

with open('probid_list.pkl', 'rb') as f:
    probid_onco_list = pickle.load(f)

print(probid_onco_list)
print("len(ILMN_list): {}".format(len(ILMN_list)))
for probid in probid_onco_list:
    if probid in ILMN_list:
        ILMN_list.remove(probid)
    else:
        print("probid {} doesn't exist in ILMN_list".format(probid))

print("len(ILMN_list) after removing oncotypes prob ids: {}".format(len(ILMN_list)))

# Generate a set of random ids
n_samples = 2000 #200
random_int_set = random.sample(range(len(ILMN_list)), n_samples)

# Get the corresponding "ILMN_*" ids
probid_list = [ILMN_list[i] for i in random_int_set]

print(probid_list)
with open('probid_list_metabric_random_subset_plus_oncotype_'+str(n_samples)+'.pkl', 'wb') as f:
    pickle.dump(probid_list, f)

datapath = '~/code/data/coxtf/Metabric/metabric_exp.txt' #'/Users/yutaro/Desktop/Metabric/metabric_exp.txt'

def preprocess(r):
    temp = r[r.index.isin(probid_list)]
    return temp

tic = time.time()

reader = pd.read_csv(datapath, sep='\t', chunksize=50)

df = pd.concat((preprocess(r) for r in reader), ignore_index=False)

print(df.shape)

print('Time: {}'.format(time.time() - tic))

df.to_csv('./metabric_random_subset_plus_oncotype_'+str(n_samples))
