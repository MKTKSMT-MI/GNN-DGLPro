import dgl
from dgl.data import DGLDataset
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
from dgl.nn import GraphConv,MaxPooling
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import time
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os
import yaml
import time
import datetime


class STL10TrainDataset(DGLDataset):
    def __init__(self,data_path,transforms=None):
        self.data_path = data_path
        self.transforms = transforms
        super().__init__(name='stl10_train_gprah')
    
    def process(self):
        GRAPHS, LABELS = dgl.load_graphs(self.data_path) #保存したグラーフデータの読み込み
        self.graphs = GRAPHS #グラフリストを代入
        self.labels = LABELS['label'] #ラベル辞書の値のみ代入
        self.dim_nfeats=len(self.graphs[0].ndata['f'][0])

    def __getitem__(self, idx):
        if self.transforms == None:
            return self.graphs[idx], self.labels[idx]
        else:
            data=self.transforms(self.graphs[idx])
            return data,self.labels[idx]
    def __len__(self):
        return len(self.graphs)


class STL10TestDataset(DGLDataset):
    def __init__(self,data_path,transforms=None):
        self.data_path = data_path
        self.transforms = transforms
        super().__init__(name='stl10_test_gprah')
    
    def process(self):
        GRAPHS, LABELS = dgl.load_graphs(self.data_path) #保存したグラーフデータの読み込み
        self.graphs = GRAPHS #グラフリストを代入
        self.labels = LABELS['label'] #ラベル辞書の値のみ代入
        self.dim_nfeats=len(self.graphs[0].ndata['f'][0])

    def __getitem__(self, idx):
        if self.transforms == None:
            return self.graphs[idx], self.labels[idx]
        else:
            data=self.transforms(self.graphs[idx])
            return data,self.labels[idx]
        
    def __len__(self):
        return len(self.graphs)


class PatchGCN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,linear=False):
        super(PatchGCN,self).__init__()
        self.linear_on = True
        self.input_layer=GraphConv(input_size,hidden_size[0])
        self.middle_layers=nn.ModuleList([GraphConv(hidden_size[i],hidden_size[i+1]) for i in range(len(hidden_size)-1)])
        self.output_layer=GraphConv(hidden_size[-1],output_size)
        if self.linear_on:
            self.linear_layers=nn.ModuleList([nn.Linear(hidden_size[i],hidden_size[i]) for i in range(len(hidden_size))])
        self.m=nn.LeakyReLU()

        self.flatt=nn.Flatten()

    
    def forward(self,g,n_feat,e_feat=None):
        n_feat=self.flatt(n_feat)
        h=self.input_layer(g,n_feat,None,e_feat).clamp(0)
        for i,layer in enumerate(self.middle_layers):
            if self.linear_on:
                h=self.linear_layers[i](h)
            h=layer(g,h)
            h=self.m(h)
        h=self.output_layer(g,h).clamp(0)
        g.ndata['h'] = h

        return dgl.mean_nodes(g,'h')
    

def train(epochs,data_path,config,linear=False):
    
    pass