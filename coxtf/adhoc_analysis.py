import numpy as np
import matplotlib.pyplot as plt
import pickle 
from pandas import DataFrame 

with open("results/run-2020-01-27-08-31-49-metabric_full/run_0/stochastic_gates.pkl", "rb") as f:
    stg_data = pickle.load(f)

print(len(stg_data))
print(stg_data.shape)
print(np.mean(np.minimum(1, np.maximum(0, 0.5+stg_data[:, 250]))))

def hard_sigmoid_np(x):
    return np.minimum(1, np.maximum(0, 0.5+x))

print(np.max(hard_sigmoid_np(stg_data[:, 250])))

df = DataFrame(hard_sigmoid_np(stg_data[:,250])) 
df[(df != 0).all(1)].plot.hist(bins=100)
#plt.show()
plt.savefig('stg_frequency')

#plt.hist(hard_sigmoid_np(stg_data[:, 250]), bins=100)
#plt.show()


np.savetxt('test.log', hard_sigmoid_np(stg_data[:, 250]))
