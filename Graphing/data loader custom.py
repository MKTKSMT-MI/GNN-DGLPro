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
from torchvision.datasets import STL10
from torchvision import transforms
import torchvision.transforms.functional as F

target_size=(224,224)
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(1)
])

STL10_train = STL10("Graphing/STL10", split='train', download=True, transform=transform)
 
STL10_test = STL10("Graphing/STL10", split='test', download=True, transform=transform)

traindataloader = DataLoader(dataset=STL10_train,batch_size=32,shuffle=True)

for image,label in traindataloader:
    print(image.shape)
    break