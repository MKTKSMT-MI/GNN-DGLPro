import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import random
import cv2
from tqdm import tqdm
import dgl
import networkx as nx
from torch.utils.data import DataLoader
from torchvision.datasets import STL10,VOCSegmentation
from torchvision import transforms
import torchvision.transforms.functional as F

target_size=(224,224)
'''
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(1)
])
'''
STL10_train = VOCSegmentation("Graphing/VOCSeg", year='2012', image_set='train', download=True)
 
STL10_test = VOCSegmentation("Graphing/VOCSeg", year='2012', image_set='trainval', download=True)

'''traindataloader = DataLoader(dataset=STL10_train,batch_size=32,shuffle=True)

for image,label in traindataloader:
    print(image.shape)
    break'''