#!/usr/bin/env python
# coding: utf-8




import os





import pickle





random_seed = 42




import random
random.seed(random_seed)
import numpy as np
np.random.seed(random_seed)

import pandas as pd
pd.set_option('display.max_rows', 512)
pd.set_option('display.max_columns', 512)
pd.set_option('display.max_colwidth', None)





import csv





import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
#get_ipython().run_line_magic('matplotlib', 'inline')
#set_matplotlib_formats('svg')





import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import Dataset





torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False





from tqdm import tqdm





from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error





from scipy import stats




def get_prob(label, prob):
    return prob[label]




output_collections_list = []
for idx in range(0, 7500+1, 2500):
    with open("saved_cifar100/score_cifar100_mix_nd0/{}.pkl".format(idx), "rb") as handle:
        output_collections = pickle.load(handle)    
        print(output_collections[0]['prob'])
        print(len(output_collections))
        
    if idx==0:
        output_collections_list = output_collections
    else:
        output_collections_list += output_collections
len(output_collections_list)        





data = []
for i in output_collections_list:
    data.append([i['index'], i['prob'], i['label'], i['prediction'],
                 i['influence_prime'], 
                 i['influence'], 
                 i['diff'], 
                 i['theta'], i['attributions'],
                ])

df_0 = pd.DataFrame(data, columns=['cifar_index', 'prob', 'label', 'prediction',
                                   'influence_prime', 'influence', 'diff', 
                                   'theta', 'attributions'
                                  ])



df_0['theta'] = -df_0['theta']
df_0['attributions'] = -df_0['attributions']



df_0['rank'] = df_0['influence'].rank(method='first', ascending=False)



mean_squared_error(df_0['influence'], df_0['theta'])**0.5



df_0['influence_prime'].plot.hist()





df_0['influence'].plot.hist()





df_0['theta'].plot.hist()






drop_index = df_0[df_0['influence']<=0].index
len(drop_index)


df_0.loc[drop_index, 'influence'] = 0.0




def post_process(theta, attributions):
    if theta>0:
        return attributions
    else:
        return np.zeros(attributions.shape)





df_0['attributions'] = df_0.apply(lambda x: post_process(x['theta'], x['attributions']), axis=1)





drop_index = df_0[df_0['theta']<=0].index
len(drop_index)





df_0.loc[drop_index, 'theta'] = 0.0





df_0.head()



df_0['rank'].head()


mem_list = []
for i in range(0, 100, 10):
    k = int(len(df_0)*i/100)
    print(k)
    mem_list.append(df_0[df_0['rank']==k+1]['influence'].values[0])





df_0_sorted = df_0.sort_values(by=['influence'], ascending=False) 
df_0_sorted[['cifar_index', 'label', 'prediction', 'prob', 'influence', 
             'rank'
            ]].head()


np.random.seed(random_seed)

total = len(df_0)
for percentage in range(10, 100, 10):
    k = int(total*(percentage/100))
    print(percentage, k)
    
    tmp = df_0_sorted.head(k)
    print(classification_report(tmp['label'], tmp['prediction']))

    tmp = df_0.drop(tmp.index)
    print(tmp['label'].value_counts())
    
    filename = "data_cifar100/mem_nd/{}.csv".format(percentage)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    tmp[['cifar_index', 'label']].to_csv(filename, index=False)



def get_random_attribution(attributions):
            
    return np.random.rand(7, 7)     




def get_mask(percentage, attributions):
    
    mask = np.zeros((7, 7))
    mask = mask.reshape(-1)
    
    length = len(mask)
    
    attributions = attributions.copy()
        
    k = int( percentage/100*(length))
    
    if k==0:
        k = 1
     
    indices = (-attributions.reshape(-1)).argsort()[:k] # top-k
    
    for i in indices:
        mask[i] = 1
        
    mask = mask.reshape(attributions.shape)
                
    return mask




df_attr = df_0_sorted.head(1000).copy()




df_attr.head()




df_attr['label'].value_counts()



for percentage in range(0, 100, 10):
    print(percentage)
    ####
    df_attr['mask'] = df_attr.apply(lambda x: get_mask(percentage, x['attributions']), axis=1)
    
    filename = "data_cifar100/mem_nd/{}.pkl".format(percentage)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    
    with open(filename, "wb") as handle:
        pickle.dump(df_attr['mask'].values.tolist(), handle)  



'''
np.random.seed(0)
df_attr['random_attributions'] = df_attr.apply(lambda x: get_random_attribution(x['attributions']), axis=1)
      
for percentage in range(0, 100, 10):
    print(percentage)  
    ####
    df_attr['mask'] = df_attr.apply(lambda x: get_mask(percentage, x['random_attributions']), axis=1)
  
    filename = "data/random_0/{}.pkl".format(percentage)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    
    with open(filename, "wb") as handle:
        pickle.dump(df_attr['mask'].values.tolist(), handle)  





np.random.seed(2)
df_attr['random_attributions'] = df_attr.apply(lambda x: get_random_attribution(x['attributions']), axis=1)
      
for percentage in range(0, 100, 10):
    print(percentage)  
    ####
    df_attr['mask'] = df_attr.apply(lambda x: get_mask(percentage, x['random_attributions']), axis=1)
  
    filename = "data/random_2/{}.pkl".format(percentage)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    
    with open(filename, "wb") as handle:
        pickle.dump(df_attr['mask'].values.tolist(), handle)  

'''



np.random.seed(42)
df_attr['random_attributions'] = df_attr.apply(lambda x: get_random_attribution(x['attributions']), axis=1)
      
for percentage in range(0, 100, 10):
    print(percentage)  
    ####
    df_attr['mask'] = df_attr.apply(lambda x: get_mask(percentage, x['random_attributions']), axis=1)
  
    filename = "data_cifar100/random/{}.pkl".format(percentage)
    os.makedirs(os.path.dirname(filename), exist_ok=True)   
    
    with open(filename, "wb") as handle:
        pickle.dump(df_attr['mask'].values.tolist(), handle)  


df = df_attr[[ 'cifar_index', 'label']].copy()
df['sample_index'] = range(len(df))
df.head()



df.info()



filename = "data_cifar100/attr_nd.csv".format(percentage)
os.makedirs(os.path.dirname(filename), exist_ok=True)

df.to_csv(filename, index=False)






