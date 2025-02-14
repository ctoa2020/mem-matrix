# -*- coding: utf-8 -*-
import os


import config

import random
import numpy as np
import pandas as pd
pd.set_option('max_colwidth', 256)

# +
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

# %matplotlib inline
#hsun test
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
#plt.rcParams['figure.dpi']=150

#set_matplotlib_formats('svg')
# -


import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

# +
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision import models
# -



from tqdm import tqdm
from sklearn.metrics import classification_report

import pickle
from contexttimer import Timer

from model import CustomModel
from dataset import CustomDataset

#hsun test
#os.environ['CUDA_VISIBLE_DEVICES']='0'


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    opt = config.parse_opt_if_attr()
    print(opt)
    ####
    set_seeds(opt.SEED)
    ####
    input_size = 224

    transform_test = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize(mean=(0.4467, 0.4398, 0.4066), std=(0.2603, 0.2566, 0.2713))
    ])
    ####
    print(os.path.join(opt.DATA_PATH, opt.ORDER, '{}.csv'.format(opt.PERCENTAGE)))
    train = pd.read_csv(os.path.join(opt.DATA_PATH, opt.ORDER, '{}.csv'.format(opt.PERCENTAGE)))
    print(train.info())

    train_dev = pd.read_csv(os.path.join(opt.DATA_PATH, opt.TRAIN_DEV_DATA))
    # train_dev = pd.read_csv(os.path.join(opt.DATA_PATH, 'pred_corr_cifar100.csv'))
    # dev = pd.read_csv(os.path.join(opt.DATA_PATH, opt.DEV_DATA))
    ####
    #train_dataset = CustomDataset(train, root=f'./data/cifar_10/train_{opt.BETA}/')

    #train_dataset = CustomDataset(train, root=f'./data_stl10/stl10/train/')
    #train_dataset = CustomDataset(train, root=f'./data_cifar100/cifar100/train_{opt.BETA}/')
    #train_dataset = CustomDataset(train, root=f'./data_cifar100/cifar100/train/')
    ####
    # modifed train dev dataset

    train_dev_dataset = CustomDataset(data=train_dev, root=f'./data/cifar_10/train/')
    train_dev_dataloader = DataLoader(train_dev_dataset, batch_size=1,
                                      shuffle=False,
                                      #                                                pin_memory=True,
                                      num_workers=0)

    #train_dev_dataset = CustomDataset(data=train_dev, root=f'./data_stl10/stl10/train/')
    #train_dev_dataset = CustomDataset(data=train_dev, root=f'./data_cifar100/cifar100/train_{opt.BETA}/')
    #train_dev_dataset = CustomDataset(data=train_dev, root=f'./data_cifar100/cifar100/train/')
    #train_dev_dataloader = DataLoader(train_dev_dataset, batch_size=1,
    #                                  shuffle=False,
                                      #                                                pin_memory=True,
    #                                  num_workers=0)
    '''
    dev_dataset = CustomDataset(dev, root=f'./data/cifar_100/train_{opt.BETA}/')
    dev_dataloader = DataLoader(dev_dataset, batch_size=opt.TEST_BATCH_SIZE, 
                                         shuffle=False, 
#                                          pin_memory=True, 
                                         num_workers=0)
    '''
    ####
    model = CustomModel(opt)
    for name, param in model.named_parameters():
        print(name)
        print(param.requires_grad)
    model.cuda()

    filename = os.path.join(opt.OUTPUT, opt.LAYER1, 'cifar10_mix_nd', str(opt.CHECKPOINT),'checkpoint/{}.pth'.format(opt.EPOCH - 1))
    #filename = os.path.join(opt.OUTPUT, opt.LAYER1, str(opt.PERCENTAGE), str(opt.CHECKPOINT),'checkpoint/{}.pth'.format(opt.EPOCH - 1))
    model.load_state_dict(torch.load(filename))

    no_decay = ['bias', 'LayerNorm.weight']

    model.cuda()
    model.eval()
    ####

    predictions_list = []
    '''
    for idx, batch in enumerate(tqdm(dev_dataloader)):

        labels = batch[1].cuda()
        inputs = batch[2].cuda()

        with torch.no_grad():
            outputs = model(inputs)    

            predictions = outputs.detach().cpu().numpy()
            predictions_list.append(predictions)

    predictions = np.vstack(predictions_list)
    predictions = np.argmax(predictions, axis=1)
    dev['prediction'] = predictions

    print(classification_report(dev['label'], dev['prediction'], digits=4))  
    '''
    ####
    start = opt.START
    length = opt.LENGTH
    print(start, length)
    start=0
    length=10000
    output_collections = []

    # modified by hsun 20240906
    # noise_inputs = (torch.rand(1,2048,7,7)).to('cuda')
    ####
    L=[]
    for idx, batch in enumerate(train_dev_dataloader):
        ####
        if idx < start:
            continue
        if idx >= start + length:
            break
        ####
        z_index = batch[0]
        z_labels = batch[1].cuda()
        z_inputs = batch[2].cuda()

        baseline = torch.zeros_like(z_inputs)
        # modified by hsun 20240906
        # beta = 0.05
        # new_inputs = z_inputs + beta * noise_inputs
        ####
        outputs = model(z_inputs)
        # outputs = model(new_inputs)
        ####
        prob = F.softmax(outputs, dim=-1)
        prediction = torch.argmax(prob, dim=1)


        if z_labels == prediction:
            L.append((z_index.item(),z_labels.item(),prediction.item()))
    df_L = pd.DataFrame(L, columns=['cifar_index', 'label', 'prediction'])
    #df_L.to_csv(f'saved_stl10/stl10_{opt.BETA}.csv', index=False)
    df_L.to_csv(f'data/cifar_pred.csv', index=False)