import numpy as np
import pickle 
#data_preprocessing/probid_list_metabric_random_subset_plus_oncotype_2000.pkl
with open("probid_list_metabric_random_subset_plus_oncotype_2000.pkl", "rb") as f:
    data = pickle.load(f)
print(data)
