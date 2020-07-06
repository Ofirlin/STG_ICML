import pandas as pd
import numpy as np
import pickle
datapath = '/Users/yutaro/Desktop/Metabric/metabric_exp.txt'

reader = pd.read_csv(datapath, sep='\t', chunksize=50)

#df = pd.concat((r.iloc[:,0] for r in reader), ignore_index=False)

save_list = []
for r in reader:
    print(type(r))
    save_list += list(r.index)

#ilmn_np_array = np.array(save_list).reshape(-1)
#np.save('ILMN_list', ilmn_np_array)
with open('ILMN_list.pkl', 'wb') as f:
    pickle.dump(save_list, f)
