import matplotlib
import matplotlib.pyplot as plt
import pickle
import random

from scipy.ndimage import zoom
from torchvision.transforms.v2.functional import to_pil_image

random.seed(42)
import numpy as np
np.random.seed(42)
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

#while True:
rate = 'uni_1.0'
beta =1.0
#input_index=42767
input_index=45624

output_collections_list = []
for idx in range(0, 5400 + 1, 1800):
    with open("saved/score_cifar10_mix_{}/{}.pkl".format(rate,idx), "rb") as handle:
        output_collections = pickle.load(handle)


    if idx == 0:
        output_collections_list = output_collections
    else:
        output_collections_list += output_collections

'''
with open("saved_stl10/score_sr42/{}.pkl".format(0), "rb") as handle:
    output_collections_list = pickle.load(handle)
'''
data = []
for i in output_collections_list:
    data.append([i['index'], i['prob'], i['label'], i['prediction'],
                 i['influence_prime'], i['influence'], i['diff'],
                 i['theta'], i['attributions'],
                 ])

df_0 = pd.DataFrame(data, columns=['cifar_index', 'prob', 'label', 'prediction',
                                   'influence_prime', 'influence', 'diff',
                                   'theta', 'attributions'
                                   ])
df_0['attributions'] = -df_0['attributions']

for _,data in df_0.iterrows():
    if data['cifar_index']==input_index:
        outputs=data
        break

input_size = 224
transform_test = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #transforms.Normalize(mean=(0.4467, 0.4398, 0.4066), std=(0.2603, 0.2566, 0.2713))
])

train_dataset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_test)
#train_dataset_full = torchvision.datasets.STL10(root='./data_stl10', split='train', download=False,transform=transform_test)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def unnormalize(img, mean, std):
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)  # unnormalize
    npimg = img.numpy()
    npimg = np.squeeze(npimg, axis=0)
    npimg = np.transpose(npimg, (1, 2, 0))

    return npimg


from matplotlib.colors import TwoSlopeNorm



#import cv2
#from numpy import asarray

#search picture
fig = plt.figure(figsize=(16, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
#ax3 = fig.add_subplot(133)

#add noise

noise_input = torch.load('data/noise/nd0.2.pt')

new_image = (train_dataset_full[outputs['cifar_index']][0].to('cuda')+beta * noise_input).cpu()
original_image = unnormalize(new_image, mean, std)

#original_image = unnormalize(train_dataset_full[outputs['cifar_index']][0], mean, std)
print(original_image.shape)
attributions = zoom(outputs['attributions'], (32, 32), order=2)

ax1.imshow(original_image)
ax1.set_axis_off()

norm = TwoSlopeNorm(vmin=min(attributions.min(),
                             -(attributions.min())),
                    vcenter=0,
                    vmax=max(attributions.max(),
                             -(attributions.max()))
                    )
Y = 0.299 * original_image[:, :, 0] + 0.587 * original_image[:, :, 1] + 0.114 * original_image[:, :, 2]
ax2.imshow(Y, cmap='gray', vmin=0, vmax=1, alpha=1)
attr_img = ax2.imshow(attributions,
                              #                          cmap=cmap,
                              #                          vmin=-0.8, vmax=0.8,
                              cmap='RdBu_r',
                              norm=norm,
                              alpha=0.8)

#circle = plt.Circle((112, 112), 112, color='white', fill=False)
#ax2.add_patch(circle)
ax2.set_axis_off()
'''
noise_input = np.squeeze(noise_input, axis=0).cpu()
pic = to_pil_image(noise_input)
noise_input = np.transpose(pic, (1, 2, 0))
ax3.imshow(pic)
ax3.set_axis_off()
'''
plt.tight_layout()


plt.savefig(f'./saved/vis_new/0_test{rate}.png')


