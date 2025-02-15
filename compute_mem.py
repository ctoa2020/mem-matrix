import os
import config
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import matplotlib_inline
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import classification_report
import pickle
from contexttimer import Timer
from model import CustomModel
from dataset import CustomDataset


pd.set_option('max_colwidth', 256)

matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_s(model,
              v,
              train_data_loader, 
              damp,
              scale,
              num_samples):
    
    last_estimate = list(v).copy()
    with tqdm(total=num_samples) as pbar:
        for i, batch in enumerate(train_data_loader):
            ####
            labels = batch[1].cuda()
            inputs = batch[2].cuda()
            ####
            this_estimate = compute_hessian_vector_products(model=model,
                                                            vectors=last_estimate,
                                                            labels=labels,
                                                            inputs=inputs,
                                                           )
            # Recursively caclulate h_estimate
            with torch.no_grad():
                new_estimate = [
                    a + (1 - damp) * b - c / scale 
                    for a, b, c in zip(v, last_estimate, this_estimate)
                ]
            ####    
            pbar.update(1)
            
            new_estimate_norm = new_estimate[0].norm().item()
            last_estimate_norm = last_estimate[0].norm().item()
            estimate_norm_diff = new_estimate_norm - last_estimate_norm
            pbar.set_description(f"{new_estimate_norm:.2f} | {estimate_norm_diff:.2f}")
            ####
            last_estimate = new_estimate
            
            if i > num_samples: # should be i>=(num_samples-1) but does not matters
                break
                
    inverse_hvp = [X / scale for X in last_estimate]
    
    return inverse_hvp


def compute_hessian_vector_products(model,
                                    vectors,
                                    labels,
                                    inputs):
    ####
    outputs = model(inputs)
    ce_loss = F.cross_entropy(outputs, labels)
    ####
    hack_loss = torch.cat([
        (p**2).view(-1)
        for n, p in model.named_parameters()
        if ((not any(nd in n for nd in no_decay)) and (p.requires_grad==True))
    ]).sum() * (opt.L2_LAMBDA)     
    ####
    loss = ce_loss + hack_loss
    ####
    model.zero_grad()
    grad_tuple = torch.autograd.grad(
        outputs=loss,
        inputs=[param for name, param in model.named_parameters()
                if param.requires_grad], 
        create_graph=True)
    ####
    #model.zero_grad()
    grad_grad_tuple = torch.autograd.grad(
        outputs=grad_tuple,
        inputs=[param for name, param in model.named_parameters() 
                if param.requires_grad],
        grad_outputs=vectors,  ##important
        only_inputs=True
    )

    return grad_grad_tuple


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
    #print(os.path.join(opt.DATA_PATH, opt.ORDER, '{}.csv'.format(opt.PERCENTAGE)))



    train = pd.read_csv(os.path.join(opt.DATA_PATH, opt.ORDER, '{}.csv'.format(opt.PERCENTAGE)))
    print(train.info())

    train_dev = pd.read_csv(os.path.join(opt.DATA_PATH, opt.TRAIN_DEV_DATA))
    #train_dev = pd.read_csv(os.path.join(opt.DATA_PATH, 'pred_corr_cifar100.csv'))
    #dev = pd.read_csv(os.path.join(opt.DATA_PATH, opt.DEV_DATA))
    ####
    #train_dataset = CustomDataset(train, root=f'./data_stl10/stl10/train/')
    train_dataset = CustomDataset(train, root=f'./data_cifar100/cifar100/train_{opt.BETA}/')
    #train_dataset = CustomDataset(train, root=f'./data_cifar100/cifar100/train/')
    ####
    #modifed train dev dataset
    '''
    train_dev_dataset = CustomDataset(data=train_dev, root='./data_cifar100/cifar100/train/')
    train_dev_dataloader = DataLoader(train_dev_dataset, batch_size=1,
                                      shuffle=False,
                                      #                                                pin_memory=True,
                                      num_workers=0)
    '''
    train_dev_dataset = CustomDataset(data=train_dev, root=f'./data_cifar100/cifar100/train_{opt.BETA}/')
    #train_dev_dataset = CustomDataset(data=train_dev, root=f'./data_cifar100/cifar100/train/')
    #train_dev_dataset = CustomDataset(data=train_dev, root=f'./data_stl10/stl10/train_{opt.BETA}/')
    train_dev_dataloader = DataLoader(train_dev_dataset, batch_size=1,
                                               shuffle=False, 
                                               #pin_memory=True, 
                                               num_workers=0)
    '''
    dev_dataset = CustomDataset(dev, root=f'./data/cifar_100/train_{opt.BETA}/')
    dev_dataloader = DataLoader(dev_dataset, batch_size=opt.TEST_BATCH_SIZE, 
                                         shuffle=False, 
                                         #pin_memory=True, 
                                         num_workers=0)
    '''
    ####
    model = CustomModel(opt)
    for name, param in model.named_parameters():
        print(name)
        print(param.requires_grad)
    model.cuda()

    filename = os.path.join(opt.OUTPUT, opt.LAYER1, 'cifar100_mix_nd', str(opt.CHECKPOINT), 'checkpoint/{}.pth'.format(opt.EPOCH-1))
    #filename = os.path.join(opt.OUTPUT, 'resnet100', str(opt.PERCENTAGE), str(opt.CHECKPOINT),'checkpoint/{}.pth'.format(opt.EPOCH - 1))
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

    output_collections = []

    # modified by hsun 20240906
    #noise_inputs = (torch.rand(1,2048,7,7)).to('cuda')
    #### 
    for idx, batch in enumerate(train_dev_dataloader):
         ####
        if idx < start:
            continue
        if idx >= start+length:
            break
    ####
        z_index = batch[0]
        z_labels = batch[1].cuda()
        z_inputs = batch[2].cuda()

        baseline = torch.zeros_like(z_inputs)
        #modified by hsun 20240906
        #beta = 0.05
        #new_inputs = z_inputs + beta * noise_inputs
    ####
        outputs = model(z_inputs)
        #outputs = model(new_inputs)
    ####
        prob = F.softmax(outputs, dim=-1)
        prediction = torch.argmax(prob, dim=1)
        
        #if prediction==z_labels:
            #continue
            
        prob_gt = torch.gather(prob, 1, z_labels.unsqueeze(1))
        #print(f'prob_gt={prob_gt}')
    ####    
        model.zero_grad()

        v = torch.autograd.grad(outputs=prob_gt, 
                                inputs=[param for name, param in model.named_parameters() 
                                        if param.requires_grad],
                                create_graph=False) 
    ####
        for repetition in range(4):      
            with Timer() as timer:
            ####
                train_dataloader = DataLoader(train_dataset, 
                                              batch_size=1, 
                                              shuffle=True, 
                                              #pin_memory=True,
                                              num_workers=0)
            ####
                s = compute_s(model=model,
                              v=v,
                              train_data_loader=train_dataloader,
                              damp=opt.DAMP, 
                              scale=opt.SCALE,
                              num_samples=opt.NUM_SAMPLES)
            ####
                time_elapsed = timer.elapsed
                #print(f"{time_elapsed:.2f} seconds")
 
        ####
            hessian = None
            steps = 50

        ####
            #baseline = torch.zeros_like(z_inputs)
            start_time = time.time()
            for alpha in np.linspace(0, 1.0, num=steps+1, endpoint=True): # right Riemann sum
                emb = baseline.clone() + alpha*(z_inputs.clone()-baseline.clone())
                #modified by hsun 20240906
                #emb = baseline.clone() + alpha * (new_inputs.clone() - baseline.clone())
                emb.requires_grad = True
            ####
                outputs = model(emb)
            ####
                ce_loss_gt = F.cross_entropy(outputs, z_labels)
                z_hack_loss = torch.cat([
                (p**2).view(-1)
                for n, p in model.named_parameters()
                if ((not any(nd in n for nd in no_decay)) and (p.requires_grad==True))
                ]).sum() * (opt.L2_LAMBDA)
            ####
                model.zero_grad()          


                grad_tuple_ = torch.autograd.grad(outputs=ce_loss_gt+z_hack_loss, 
                                                  inputs=[param for name, param in model.named_parameters() 
                                                          if param.requires_grad],
                                                  create_graph=True)   
            
                #model.zero_grad()
                my_s = nn.utils.parameters_to_vector(s).detach()
                #my_grad_tuple_ = nn.utils.parameters_to_vector(grad_tuple_)
                dot = nn.utils.parameters_to_vector(s).detach() @ nn.utils.parameters_to_vector(grad_tuple_) # scalar
                #shape of s = 4216842*1
                #shape of H'= 4216842*(2048*7*7)
                #shape of z_inputs = 1*(2048*7*7)
                #shape of result(=-r^T(z_inputs)) = 2048*7*7  //2048 pairs of 7*7 matrices multiplications
                #shape of Mreplace(z)(=A summation of 2048 7*7 matrices) = 7*7
                grad_grad_tuple_ = torch.autograd.grad(
                    outputs=dot,
                    inputs=[emb],
                    #inputs=[z_input],
                    #we doubt that it should be z_input here instead of emb. 20240824
                    only_inputs=True
                )
            ####
                if alpha==0:
                    hessian = grad_grad_tuple_[0]
                    influence_prime = [-torch.sum(x * y) for x, y in zip(s, grad_tuple_)] 
                    influence_prime = sum(influence_prime).item()
                elif alpha==1.0:
                    break
                else:
                    hessian += grad_grad_tuple_[0]
        ####

            end_time = time.time()
            elapsed_time = end_time - start_time
            #print(f"代码片段执行时间: {elapsed_time:.4f} 秒")

            hessian = hessian / steps
        ####  
            influence = [-torch.sum(x * y) for x, y in zip(s, grad_tuple_)] 
            influence = sum(influence).item()
        ####

            #modified by hsun 20240906
            result = hessian*(z_inputs)
            #result = hessian * (new_inputs)
            result = torch.sum(result, dim=1)
            #print(result.size())

            theta = torch.sum(result).detach().cpu().numpy()
        
            outputs = {
            "index": z_index.detach().cpu().numpy()[0],
            "label": z_labels.detach().cpu().numpy()[0],
            "prob": prob.detach().cpu().numpy()[0],
            "prediction": prediction.detach().cpu().numpy()[0],
            "influence_prime": influence_prime,
            "influence": influence,
            "diff": influence_prime-influence,
            "theta": theta,
            "attributions": result[0].detach().cpu().numpy(),
            "repetition": repetition,
            "time_elapsed": time_elapsed,
            }        
            '''
            print(idx)
            print(outputs['index'])
            print(outputs['label'], outputs['prob'], outputs['prediction'])
            print('influence_prime: ', outputs['influence_prime'])
            print('influence: ', outputs['influence'])
            print('diff: ', outputs['diff'])
            print('theta: ', outputs['theta'])
            print("attributions: ", outputs['attributions'])
            print("repetition:", outputs['repetition'])
            '''
            output_collections.append(outputs)
            ####
            break


#beta for saving files
#dataset = 'cifar100'

filename = os.path.join(opt.OUTPUT, 'score_cifar100_{}/{}.pkl'.format(opt.BETA, start))
os.makedirs(os.path.dirname(filename), exist_ok=True)

with open(filename, "wb") as handle:
    pickle.dump(output_collections, handle) 



