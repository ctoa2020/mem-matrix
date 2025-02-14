#!/usr/bin/env python
# coding: utf-8




import pickle





import random

from pandas import read_csv

random.seed(42)
import numpy as np
np.random.seed(42)

import pandas as pd
pd.set_option('display.max_rows', 512)
pd.set_option('display.max_columns', 512)
pd.set_option('display.max_colwidth', None)





import csv





import matplotlib
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')


import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.utils.data import Dataset




torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False





import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision import models





from scipy.ndimage import zoom




from tqdm import tqdm



from sklearn.metrics import classification_report



from model import CustomModel
from dataset import CustomDataset




output_collections_list = []
for idx in range(0, 5400+1, 1800):
    with open("saved/score_cifar10_mix_uni_0.1/{}.pkl".format(idx), "rb") as handle:
    #with open("saved_cifar100/score_cifar100_mix_nd0/{}.pkl".format(idx), "rb") as handle:
        output_collections = pickle.load(handle)    
        print(output_collections[0]['prob'])
        print(len(output_collections))
   
    if idx==0:
        output_collections_list = output_collections
    else:
        output_collections_list += output_collections

len(output_collections_list)        
'''
output_collections_list = []
    # with open("saved/score_42/{}.pkl".format(idx), "rb") as handle:
with open("saved_stl10/score_sr42/0.pkl", "rb") as handle:
    output_collections = pickle.load(handle)
    print(output_collections[0]['prob'])
    print(len(output_collections))
output_collections_list = output_collections
len(output_collections_list)
'''


data = []
for i in output_collections_list:
    data.append([i['index'], i['prob'], i['label'], i['prediction'],
                 i['influence_prime'], i['influence'], i['diff'], 
                 i['theta'], i['attributions'],
                ])

df_0 = pd.DataFrame(data, columns=['stl10_index', 'prob', 'label', 'prediction',
                                   'influence_prime', 'influence', 'diff', 
                                   'theta', 'attributions'
                                  ])

df_0['theta'] = -(df_0['theta'].astype(float))
df_0['attributions'] = -df_0['attributions']




input_size = 224

# transform_train = transforms.Compose([
#         transforms.RandomResizedCrop(input_size),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
transform_test = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])





train_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_test)
#train_dataset_full = torchvision.datasets.STL10(root='./data_stl10', split='train', download=False,transform=transform_test)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
'''
classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
'''


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def unnormalize(img, mean, std):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m) # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    
    return npimg




import matplotlib.colors as colors
from matplotlib.colors import TwoSlopeNorm
import os



idx_list = pd.read_csv('./data/cifar_mix_uni_0.csv')





amp = idx_list.iloc[800:1200]

fig, axs = plt.subplots(20, 20, figsize=(20, 20))

idx = 0
for _,rk in amp.iterrows():

    matching_row = df_0[df_0['stl10_index'] == rk['cifar_index']]

    #if not matching_row.empty:
    outputs = matching_row.iloc[0]
    #print(f"Processing stl10_index: {outputs['stl10_index']}")
    original_image = unnormalize(train_dataset_full[outputs['stl10_index']][0], mean, std)
    attributions = zoom(outputs['attributions'], (32, 32), order=2)

    norm = TwoSlopeNorm(vmin=min(attributions.min(), -(attributions.min())),
                        vcenter=0,
                        vmax=max(attributions.max(), -(attributions.max())))

    axs[idx // 20, idx % 20].imshow(original_image)
    axs[idx // 20, idx % 20].set_axis_off()
    '''
    Y = 0.299 * original_image[:, :, 0] + 0.587 * original_image[:, :, 1] + 0.114 * original_image[:, :, 2]
    axs[idx // 20, idx % 20].imshow(Y, cmap='gray', vmin=0, vmax=1, alpha=0.4)

    attr_img = axs[idx // 20, idx % 20].imshow(attributions,
                                                     cmap='RdBu_r',
                                                     norm=norm,
                                                     alpha=0.8)
    axs[idx // 20, idx % 20].set_axis_off()
    '''
    if idx>=400:
        break
    idx+=1

fig.tight_layout()
output_directory = 'saved/vis'
os.makedirs(output_directory, exist_ok=True)
plt.savefig(f'saved/vis/uni0.pdf', bbox_inches='tight', pad_inches=0.1)

