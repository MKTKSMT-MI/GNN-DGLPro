import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dgl.nn import GraphConv,SAGEConv,GATConv
import dgl
from dgl.data import DGLDataset
import seaborn as sns
import random
from tqdm import tqdm
from models import PatchGCN,PatchSAGE,PatchGAT


class ICPKGIDataset(DGLDataset):
    def __init__(self,data_path,transforms=None):
        self.data_path = data_path
        self.transforms = transforms
        super().__init__(name='ICPKGI_gprah')
    
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
    

def train_test_split(data,data_num):
    shuffle_data=random.sample(data,len(data))
    return shuffle_data[:-data_num], shuffle_data[-data_num:]


graphs=[[] for _ in range(5)]
dataset=ICPKGIDataset('GNN-DGLPro/data/ICPKGI/8patch_gray.dgl')
for graph,label in dataset:
    graphs[label].append(graph)


model_names=['PatchGCN','PatchSAGE','PatchGAT']
hidden_sizes=[[512],[512,256],[512,512,256],[512,256,256],[512,512,512,256],[512,512,256,256],[512,256,256,256]]


attempts_number=100
cos=nn.CosineSimilarity(2)
for model_num,model_name in enumerate(model_names):
    for hidden_size in hidden_sizes:
        model = globals()[model_name](hidden_size)
