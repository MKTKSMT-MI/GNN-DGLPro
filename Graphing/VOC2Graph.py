import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import random
import cv2
from tqdm import tqdm
import dgl
import networkx as nx
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
import torchvision.transforms.functional as F

from torchvision.io.image import read_image
from torchvision.models.segmentation import deeplabv3_resnet101,DeepLabV3_ResNet101_Weights
from torchvision.transforms.functional import to_pil_image

import modules

image=read_image('Graphing/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000256.jpg')
image=F.resize(image,size=(256,256))
image=np.array(image).transpose(1,2,0)
'''plt.imshow(image)
plt.show()'''

mask=read_image('Graphing/VOC/VOCdevkit/VOC2012/SegmentationClass/2007_000256.png')
mask=F.resize(mask,size=(256,256))
mask=np.array(mask).transpose(1,2,0)
mask=np.where(mask!=0,255,mask)
'''plt.imshow(mask,vmin=0,vmax=255)
plt.show()'''

patch_num=8
data=modules.img2patch(mask,patch_num,cmap='jet',views=False)
print(np.sum(data[25]==255)/data[25].size)
modules.objYN(data,patch_num,True)

