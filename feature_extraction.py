#!/usr/bin/env python
# coding: utf-8




import os

from PIL import Image

#from cifar.search_picture import noise_input

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0'

random_seed = 42


import pickle


import random
random.seed(random_seed)
import numpy as np
np.random.seed(random_seed)
import pandas as pd
pd.set_option('max_colwidth', 256)


import matplotlib.pyplot as plt


import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')






import torch
import torch.nn as nn
from torch.nn import functional as F





torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False





import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision import models

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
    #transforms.Normalize(mean=(0.4467, 0.4398, 0.4066), std=(0.2603, 0.2566, 0.2713))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=16)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=16)

'''
trainset = torchvision.datasets.STL10(root='./data_stl10', split='train', download=False, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=16)

testset = torchvision.datasets.STL10(root='./data_stl10', split='test', download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=16)


trainset = torchvision.datasets.CIFAR100(root='./data_cifar100', train=True, download=False, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=16)

testset = torchvision.datasets.CIFAR100(root='./data_cifar100', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=16)

'''

net = models.resnet50(pretrained=True)
net = nn.Sequential(*list(net.children())[:-2])

net.cuda()
net.eval()

print('done')





from tqdm import tqdm




random.seed(42)
np.random.seed(42)
#torch.manual_seed(42)
torch.cuda.amp.autocast(enabled=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#hsun


#modified by hsun 20240909
noise_input = torch.rand(1,3,224,224).to('cuda')
#noise = torch.randn(1,3,224,224).to('cuda')
#noise_input =  torch.load('data_cifar100/noise/cifar100_0.2.pt')
#Min-Max normalization
#noise_input = (noise - noise.min())/(noise.max()-noise.min())
#torch.save(noise_input, 'data/noise/exp2.pt')
betas =[0,0.1,0.2,0.5,0.7,1.0]
#directory = f'./data/cifar_10/train_mix/'
#os.makedirs(directory, exist_ok=True)
'''
for beta in betas:
    directory = f'./data_cifar100/cifar100/test_uni_{beta}/'
    os.makedirs(directory, exist_ok=True)
'''
directory = f'./data/cifar_10/test_uni_100/'
os.makedirs(directory, exist_ok=True)

with torch.no_grad():
    '''
    train_labels_list = []
    for  idx, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        
        inputs, labels = data
            
        inputs = inputs.cuda()        
        labels = labels.cuda()
        
        with torch.no_grad():
            features = net(inputs) # bs 512 7 7
        with open('./data/cifar_10/train/{}.npy'.format(str(idx)), 'wb') as f:
            np.save(f, features.detach().cpu().numpy())
        

        train_labels_list.append(labels)
    '''
    for beta in betas:

        train_labels_list=[]
        for idx, data in tqdm(enumerate(testloader), total=len(testloader)):
            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            new_input = inputs + beta * noise_input

            with torch.no_grad():
                features = net(new_input)  # bs 512 7 7
            with open('./data/cifar_10/test_uni_{}/{}.npy'.format(beta,str(idx)), 'wb') as f:
                np.save(f, features.detach().cpu().numpy())

            train_labels_list.append(labels)
  
 
    '''
    with torch.no_grad():
        features = net(inputs) # bs 512 7 7
    with open('./data/cifar_100/train_1.0/{}.npy'.format(str(idx)), 'wb') as f:
        np.save(f, features.detach().cpu().numpy())

    train_labels_list.append(labels)
    
   
    train_labels = torch.cat(train_labels_list, dim=0).detach().cpu().numpy() # 5000 1
'''


'''
data = []
for i in range(len(train_labels)):
    data.append([i , train_labels[i]])

df_0 = pd.DataFrame(data, columns=['cifar_index','label'])
df_0.head()

df_0.info()


df_0.to_csv('data/train.csv', index=False)


df_0 = pd.read_csv('data/train.csv')
df_0.info()



#for cifar

from sklearn.model_selection import train_test_split
_, df_0_sampled = train_test_split(df_0, test_size=0.3, 
                                   random_state=random_seed, 
                                   stratify=df_0['label'])

df_0_sampled.head()


df_0_sampled['label'].value_counts()


df_0_sampled_train, df_0_sampled_dev = train_test_split(df_0_sampled, test_size=5000, 
                                                        random_state=random_seed, 
                                                        stratify=df_0_sampled['label'])


df_0_sampled_train.to_csv('data/train_10000.csv', index=False)
'''
'''



df_0_sampled_dev.to_csv('data/dev_5000.csv', index=False)




random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

test_labels_list = []

#hsun
directory = './data_cifar100/cifar100/test/'
os.makedirs(directory, exist_ok=True)

with torch.no_grad():
    for idx, data in tqdm(enumerate(testloader), total=len(testloader)):
        inputs, labels = data
            
        inputs = inputs.cuda()        
        labels = labels.cuda()
        
        with torch.no_grad():
            features = net(inputs) # bs 512 7 7
        with open('./data_cifar100/cifar100/test/{}.npy'.format(str(idx)), 'wb') as f:
            np.save(f, features.detach().cpu().numpy())
        
        test_labels_list.append(labels)

    test_labels = torch.cat(test_labels_list, dim=0).detach().cpu().numpy() # 5000 1




data = []
for i in range(len(test_labels)):
    data.append([i, test_labels[i]])

df_1 = pd.DataFrame(data, columns=['cifar_index','label'])
df_1.head()




df_1.info()




df_1.to_csv('data_cifar100/test.csv', index=False)




with open('./data/cifar_10/test/0.npy', 'rb') as f:
    tmp = np.load(f)
tmp.shape



total = len(df_0_sampled_train)
for percentage in range(0, 100, 10):
    k = int(total*(percentage/100))
    print(percentage, k)
    
    tmp = df_0_sampled_train.sample(k, 
                       random_state=0
                      )
    
    tmp = df_0_sampled_train.drop(tmp.index)
    print(tmp['label'].value_counts())
    
    filename = "data/random_0/{}.csv".format(percentage)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    tmp[['cifar_index', 'label']].to_csv(filename, index=False)




total = len(df_0_sampled_train)
for percentage in range(0, 100, 10):
    k = int(total*(percentage/100))
    print(percentage, k)
    
    tmp = df_0_sampled_train.sample(k, 
                       random_state=2
                      )

    tmp = df_0_sampled_train.drop(tmp.index)
    print(tmp['label'].value_counts())
    
    filename = "data/random_2/{}.csv".format(percentage)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    tmp[['cifar_index', 'label']].to_csv(filename, index=False)


#delete next time
df_0_sampled_train=pd.read_csv('./data_cifar100/train_10000.csv')

total = len(df_0_sampled_train)
for percentage in range(0, 100, 10):
    k = int(total*(percentage/100))
    print(percentage, k)
    
    tmp = df_0_sampled_train.sample(k, 
                       random_state=42
                      )

    tmp = df_0_sampled_train.drop(tmp.index)
    print(tmp['label'].value_counts())
    
    filename = "data_cifar100/random/{}.csv".format(percentage)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    tmp[['cifar_index', 'label']].to_csv(filename, index=False)




#stl10 random_mix is wrong run again next time

tmp = df_0_sampled_train
print(tmp['label'].value_counts())

filename = "data_cifar10/random/{}.csv".format(0)
os.makedirs(os.path.dirname(filename), exist_ok=True)

tmp[['cifar_index', 'label']].to_csv(filename, index=False)
'''




