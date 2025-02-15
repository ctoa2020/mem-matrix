#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import classification_report
from scipy import stats
import csv
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

pd.set_option('display.max_rows', 512)
pd.set_option('display.max_columns', 512)
pd.set_option('display.max_colwidth', None)



torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False






beta = 1.0

output_collections_list = []
for idx in range(0, 5400+1, 1800):
    with open("saved/score_cifar10_mix_uni_{}/{}.pkl".format(beta,idx), "rb") as handle:
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





drop_index = df_0[df_0['influence']<=0].index
len(drop_index)





df_0.loc[drop_index, 'influence'] = 0.0





df_0['rank'] = df_0['influence'].rank(method='first', ascending=False)





df_0['rank'].head()



mem_list = []
for i in range(0, 100, 10):
    k = int(len(df_0)*i/100)
    print(k)
    mem_list.append(df_0[df_0['rank']==k+1]['influence'].values[0])





def read_report(output_path, order, percentage, seed, epoch, split):
    df = pd.read_csv(os.path.join(output_path, order, percentage, seed, "report/{}_{}.csv".format(epoch-1, split)))
    return df


original = []
for seed in [0, 1, 2, 3, 42]:
    df = read_report('saved', 'random', str(0), str(seed), 10, f'test_uni_{beta}')
    original.append(df.loc[10][1])
original




percentage_list = [
    10, 
    20, 
    30, 
    40, 
    50
]



result_dict = {}
#mem this
#121212121212121212121121212121212121212121212121212

for order in [
    'random',
    'mem'
]:
    for percentage in percentage_list:
        result_dict[order+'_'+str(percentage)] = []
        for seed in [0, 1, 2, 3, 42]:
            df = read_report('saved', order, str(percentage), str(seed), 10, f'test_uni_{beta}')
            result_dict[order+'_'+str(percentage)].append(df.loc[10][1])
'''
for order in [
    'random_0', 
    'random_2', 
    'random', 
    'mem'
]:
    for percentage in percentage_list:
        result_dict[order+'_'+str(percentage)] = []
        for seed in [0, 1, 2, 3, 42]:
            df = read_report('saved', order, str(percentage), str(seed), 10, 'test')
            result_dict[order+'_'+str(percentage)].append(df.loc[10][1])
            '''



random_mean_ = []
random_std_ = []

random_mean_.append(np.mean(original)*100)
random_std_.append(np.std(original)*100)

for percentage in percentage_list:
    result = result_dict['random_'+str(percentage)]# + result_dict['random_0_'+str(percentage)] + result_dict['random_2_'+str(percentage)]
    random_mean_.append(np.mean(result)*100)
    random_std_.append(np.std(result)*100)



mem_mean = []
mem_std = []

mem_mean.append(np.mean(original)*100)
mem_std.append(np.std(original)*100)

for percentage in percentage_list:
    result = result_dict['mem_'+str(percentage)]
    mem_mean.append(np.mean(result)*100)
    mem_std.append(np.std(result)*100)



'''

matplotlib.rcParams.update({'font.size': 24})

fig, ax0 = plt.subplots(figsize=(8, 4))
x = range(0, 60, 10)
ax0.errorbar(
    x,
    random_mean_,
    yerr=random_std_,
    linestyle='-',
    fmt='o',
    label='random',
    color='tab:green',
    capsize=5
)

ax0.set_ylabel('Test Accuracy (%)')
ax0.set_xticks(x)
ax0.set_yticks(
    np.arange(
        np.round(min(random_mean_), 0),
        np.round(max(random_mean_), 0) + 1.0,
        0.5
    )
)

ax0.grid(True)
ax0.legend()
plt.tight_layout()
plt.show()
filename = "saved/vis_new/Marginal_CIFAR_random.pdf"
os.makedirs(os.path.dirname(filename), exist_ok=True)

plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
'''

matplotlib.rcParams.update({'font.size': 14})

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1]})

x = range(0, 60, 10)

ax0.errorbar(x, random_mean_, 
            yerr=random_std_, 
            linestyle='-', 
            fmt='o', label='random', color='tab:blue', capsize=5)
ax0.errorbar(x, mem_mean, 
            yerr=mem_std, 
            linestyle='-', fmt='o', label='memorized', color='tab:red', capsize=5)
ax0.set_title(f'Uniform Dist. with Noise Intensity {beta}',fontsize=15)
ax0.set_ylabel('Test Accuracy (%)')
ax0.set_xticks(x)

ax0.set_yticks(np.arange(np.round(min(random_mean_+mem_mean), 0), np.round(max(random_mean_+mem_mean), 0)+1.0, 0.5))

ax0.grid(True)
ax0.tick_params(axis='y', labelsize=10)
ax0.legend(loc='lower left')


ax1.set_xlabel('Fraction Removed from Train Data(%)')
ax1.set_ylabel('Mem Value')
ax1.set_xticks(x)
ax1.set_yticks(np.arange(0, df_0['influence'].max()+5, 5))




ax1.plot(x, mem_list[0: len(x)], color='tab:green', marker='.')
ax1.grid(True)

fig.tight_layout()
# plt.show()

filename = f"saved/vis_new/Marginal_CIFAR_uni_{beta}.pdf"
os.makedirs(os.path.dirname(filename), exist_ok=True)  

plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)  






'''



result_dict = {}

for order in [
    'random', 
    'mem_reverse'
]:
    for percentage in percentage_list:
        result_dict[order+'_'+str(percentage)] = []
        for seed in [0, 1, 2, 3, 42]:
            df = read_report('saved', order, str(percentage), str(seed), 10, 'test')
            result_dict[order+'_'+str(percentage)].append(df.loc[10][1])





mem_mean = []
mem_std = []

mem_mean.append(np.mean(original)*100)
mem_std.append(np.std(original)*100)

for percentage in percentage_list:
    result = result_dict['mem_reverse_'+str(percentage)]
    mem_mean.append(np.mean(result)*100)
    mem_std.append(np.std(result)*100)





matplotlib.rcParams.update({'font.size': 24})

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8), gridspec_kw={'height_ratios': [2, 1]})

x = range(0, 60, 10)

ax0.errorbar(x, random_mean_, 
            yerr=random_std_, 
            linestyle='-', 
            fmt='o', label='random', color='tab:green', capsize=5)
ax0.errorbar(x, mem_mean, 
            yerr=mem_std, 
            linestyle='-', fmt='o', label='memorized', color='tab:orange', capsize=5)

ax0.set_ylabel('Test Accuracy (%)')
ax0.set_xticks(x)

ax0.set_yticks(np.arange(np.round(min(random_mean_+mem_mean), 0), np.round(max(random_mean_+mem_mean), 0)+1.0, 0.5))

ax0.grid(True)


ax1.set_xlabel('Fraction Removed (%)')
ax1.set_ylabel('Mem Value')
ax1.set_xticks(x)
ax1.set_yticks(np.arange(0, df_0['influence'].max()+5, 5))



ax1.plot(x, mem_list[0: len(x)], color='tab:blue', marker='.')
ax1.grid(True)

fig.tight_layout()
# plt.show()

filename = "saved/vis/Marginal_CIFAR_mem_reverse.pdf"
os.makedirs(os.path.dirname(filename), exist_ok=True)  

plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)  

'''





