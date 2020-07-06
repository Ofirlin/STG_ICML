import pandas as pd
import time 
# NOTE: '/Users/yutaro/Desktop/Metabric/metabric_exp.txt' is from Prof. Kluger's Box folder.
datapath = '/Users/yutaro/Desktop/Metabric/metabric_exp.txt'

#probid_list = ['ILMN_1725881', 'ILMN_1818577'] 
# The conversion is based on http://www.genomequebec.mcgill.ca/compgen/integrated_vervet_genomics/transcriptome/Illumina/index.html
hugo_to_prob = {'ESR1':['ILMN_1678535'],'PGR':['ILMN_1811014'],
        'BCL2':['ILMN_1701120','ILMN_1697970','ILMN_1801119'], 
        'SCUBE2':['ILMN_1684085'], 'MKI67':['ILMN_1734827'], 'AURKA':['ILMN_1680955'],
        'BIRC5': ['ILMN_1710082'], 'CCNB1':['ILMN_1712803'], 'MYBL2':['ILMN_1709020'],
        'ERBB2' :['ILMN_1717902'], 'GRB7' : ['ILMN_1798582','ILMN_1740762'],
        'MMP11' : ['ILMN_1749226'], 'CTSL2':['ILMN_1748352'], 
        'GSTM1' : ['ILMN_1762255', 'ILMN_1668134'], 'CD68': ['ILMN_1714861'],
        'BAG1' : ['ILMN_1733970']}
hugo_list = ['ESR1', 'PGR', 'BCL2', 'SCUBE2', 'MKI67', 'AURKA', 'BIRC5', 'CCNB1', 
        'MYBL2', 'ERBB2', 'GRB7', 'MMP11', 'CTSL2', 'GSTM1', 'CD68', 'BAG1']


probid_list = []
for hugo in hugo_list:
    temp = hugo_to_prob[hugo]
    if len(temp) == 1:
        probid_list.append(temp[0])
    else:
        for ilmn in temp:
            probid_list.append(ilmn)

print(probid_list)

def preprocess(r):
    temp = r[r.index.isin(probid_list)]
    # print(temp)
    return temp

tic = time.time()

reader = pd.read_csv(datapath, sep='\t', chunksize=50)

df = pd.concat((preprocess(r) for r in reader), ignore_index=False)

print(df.shape)
# print(df.head())

print('Time: {}'.format(time.time() - tic))

df.to_csv('./metabric_oncotype')
