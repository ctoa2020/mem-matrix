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

from transformers import get_linear_schedule_with_warmup



from tqdm import tqdm
from sklearn.metrics import classification_report

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



# +
if __name__ == '__main__':
    opt = config.parse_opt()
    print(opt)
    set_seeds(opt.SEED)
    ####
    input_size = 224

    transform_test = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        #transforms.Normalize(mean=(0.4467, 0.4398, 0.4066), std=(0.2603, 0.2566, 0.2713))
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])    
    ####
    #print(os.path.join(opt.DATA_PATH, opt.ORDER, '{}.csv'.format(opt.PERCENTAGE)))
    train = pd.read_csv(os.path.join(opt.DATA_PATH, opt.ORDER, '{}.csv'.format(opt.PERCENTAGE)))
    #train = pd.read_csv(os.path.join(opt.DATA_PATH, 'random', '{}.csv'.format(opt.PERCENTAGE)))
    print(train.info())

    #train_dev = pd.read_csv(os.path.join(opt.DATA_PATH, opt.TRAIN_DEV_DATA))
    train_dev = pd.read_csv(os.path.join(opt.DATA_PATH,'train_10000.csv'))

    #dev = pd.read_csv(os.path.join(opt.DATA_PATH, opt.DEV_DATA))

    #cifar_10 or cifar_100
    #data_folders = ['train', 'train_mix_uni_0.1', 'train_mix_uni_0.2', 'train_mix_uni_0.5', 'train_mix_uni_0.7','train_mix_uni_1.0']
    data_folders = ['train', 'train_mix_nd0.1', 'train_mix_nd0.2', 'train_mix_nd0.5', 'train_mix_nd0.7','train_mix_nd1.0']

    #stl_10
    #data_folders = ['train', 'train_mix_0.1', 'train_mix_0.2', 'train_mix_0.5', 'train_mix_0.7','train_mix_1.0']
    #data_folders = ['train', 'train_mix_uni_0.1', 'train_mix_uni_0.2', 'train_mix_uni_0.5', 'train_mix_uni_0.7', 'train_mix_uni_1.0']


    for folder in data_folders:
    ####
        train_dataset = CustomDataset(train, root=f'./data_cifar100/cifar100/{folder}/')#10000
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.TRAIN_BATCH_SIZE,
                                              shuffle=True,
    #                                           pin_memory=True,
                                              num_workers=0)
        ####

        train_dev_dataset = CustomDataset(data=train_dev, root=f'./data_cifar100/cifar100/{folder}/')#10000
        train_dev_dataloader = torch.utils.data.DataLoader(train_dev_dataset, batch_size=opt.TEST_BATCH_SIZE,
                                                   shuffle=False,
    #                                                pin_memory=True,
                                                   num_workers=0)
        ####
        #stl10 no dev
        '''
        dev_dataset = CustomDataset(dev, root='./data_stl10/stl10/train/')#5000
        dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=opt.TEST_BATCH_SIZE,
                                             shuffle=False,
    #                                          pin_memory=True,
                                             num_workers=0)
        ####
        '''

        total_steps = len(train_dataloader) * opt.EPOCH
        print("total_steps: {}".format(total_steps))
        num_warmup_steps = int(total_steps * 0.0)
        print("num_warmup_steps: {}".format(num_warmup_steps))
        ####
        model = CustomModel(opt)
        for name, param in model.named_parameters():
            print(name)
            print(param.requires_grad)
        model.cuda()
        ####
        criterion = nn.CrossEntropyLoss()

        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters()
                        if (p.requires_grad==True)]},
        ]
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=opt.LEARNING_RATE, momentum=opt.MOMENTUM)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)
        ####
        for i in range(opt.EPOCH):  # loop over the dataset multiple times
            #print('epoch: {}'.format(i))
            model.train()
            #print(len(train_dataloader))
            for idx, batch in enumerate(tqdm(train_dataloader)):
               # zero the parameter gradients
                model.zero_grad()
                optimizer.zero_grad()

                # get the inputs
                cifar_idxes, labels, features = batch

                features = features.cuda()
                labels = labels.cuda()

                # forward + backward + optimize
                outputs = model(features)
                ce_loss = criterion(outputs, labels)

                hack_loss = torch.cat([
                    (p**2).view(-1)
                    for n, p in model.named_parameters()
                    if ((not any(nd in n for nd in no_decay)) and (p.requires_grad==True))
                ]).sum() * (opt.L2_LAMBDA)

                loss = ce_loss + hack_loss

                loss.backward()
                optimizer.step()
                scheduler.step()

    #             break
            ####
            if opt.SAVE_CHECKPOINT:
                #filename = os.path.join(opt.OUTPUT, opt.LAYER1, 'cifar10_uni', str(opt.SEED), 'checkpoint/{}.pth'.format(i))
                filename = os.path.join(opt.OUTPUT, opt.ORDER, str(opt.PERCENTAGE), str(opt.SEED),'checkpoint/{}.pth'.format(i))
                os.makedirs(os.path.dirname(filename), exist_ok=True)

                torch.save(model.state_dict(), filename)
            ####
            model.eval()
            prob_list = []
            predictions_list = []

            for idx, batch in enumerate(tqdm(train_dev_dataloader)):
                cifar_idxes, labels, features = batch

                features = features.cuda()
                labels = labels.cuda()

                with torch.no_grad():
                    outputs = model(features)

                    prob = F.softmax(outputs, dim=-1)
                    prob = torch.gather(prob, 1, labels.unsqueeze(1))
                    prob_list.append(prob.detach().cpu().numpy())

                    predictions = outputs.detach().cpu().numpy()
                    predictions_list.append(predictions)
            probs = np.vstack(prob_list)
            #print(probs[0: 5])
            predictions = np.vstack(predictions_list)
            predictions = np.argmax(predictions, axis=1)
            train_dev['prediction'] = predictions

            report = classification_report(train_dev['label'], train_dev['prediction'], digits=4, output_dict=True)
            #print(report)
            report_df = pd.DataFrame(report).transpose()

            filename = os.path.join(opt.OUTPUT, opt.ORDER, str(opt.PERCENTAGE), str(opt.SEED), 'report/{}_{}.csv'.format(i, 'train_dev'))
            #filename = os.path.join(opt.OUTPUT, opt.LAYER1, 'cifar10_mix_uni_2', str(opt.SEED),'report/{}_{}.csv'.format(i, 'train_dev'))
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            report_df.to_csv(filename)

        ####
        '''
        model.eval()
        predictions_list = []
    
        
        for idx, batch in enumerate(tqdm(dev_dataloader)):
            cifar_idxes, labels, features = batch
            
            features = features.cuda()
            labels = labels.cuda()
            
            with torch.no_grad():
                outputs = model(features)
                            
                predictions = outputs.detach().cpu().numpy()
                predictions_list.append(predictions)
                
        predictions = np.vstack(predictions_list)
        predictions = np.argmax(predictions, axis=1)
        dev['prediction'] = predictions
        
        report = classification_report(dev['label'], dev['prediction'], digits=4, output_dict=True)
        #print(report)
        report_df = pd.DataFrame(report).transpose()
    
        #filename = os.path.join(opt.OUTPUT, opt.LAYER1, 'cifar100', str(opt.SEED), 'report/{}_{}.csv'.format(i, 'dev'))
        filename = os.path.join(opt.OUTPUT, opt.ORDER, str(opt.PERCENTAGE), str(opt.SEED), 'report/{}_{}.csv'.format(i, 'dev'))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        report_df.to_csv(filename)
        ###
#         break
        '''
    test = pd.read_csv(os.path.join(opt.DATA_PATH, opt.TEST_DATA))
    test_dataset = CustomDataset(data=test, root='./data_cifar100/cifar100/test/')
    test_dataloader = DataLoader(test_dataset, batch_size=opt.TEST_BATCH_SIZE,
        num_workers=0,
        shuffle=False,
#         pin_memory=True
        )
    ####
    model.eval()

    predictions_list = []
    for idx, batch in enumerate(tqdm(test_dataloader)):

        cifar_idxes, labels, features = batch

        features = features.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            outputs = model(features)

            predictions = outputs.detach().cpu().numpy()
            predictions_list.append(predictions)

    predictions = np.vstack(predictions_list)
    predictions = np.argmax(predictions, axis=1)
    test['prediction'] = predictions

    report = classification_report(test['label'], test['prediction'], digits=4, output_dict=True)
    #print(report)
    report_df = pd.DataFrame(report).transpose()

    #filename = os.path.join(opt.OUTPUT, opt.LAYER1, str(opt.PERCENTAGE), str(opt.SEED), 'report/{}_{}.csv'.format(i, 'test'))
    filename = os.path.join(opt.OUTPUT, opt.ORDER, str(opt.PERCENTAGE), str(opt.SEED),'report/{}_{}.csv'.format(i, 'test'))
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    report_df.to_csv(filename)
    ####
