import dgl
from dgl.data import DGLDataset
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
from dgl.nn import GraphConv
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import itertools
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import os

#トレーニング用データセットクラス
class CIFAR10TrainDataset(DGLDataset):
    def __init__(self,data_path):
        self.data_path = data_path
        super().__init__(name='cifar10_train__gprah')
    
    def process(self):
        GRAPHS, LABELS = dgl.load_graphs(self.data_path) #保存したグラーフデータの読み込み
        self.graphs = GRAPHS #グラフリストを代入
        self.labels = LABELS['label'] #ラベル辞書の値のみ代入

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


class CIFAR10TestDataset(DGLDataset):
    def __init__(self,data_path):
        self.data_path = data_path
        super().__init__(name='cifar10_test_gprah')
    
    def process(self):
        GRAPHS, LABELS = dgl.load_graphs(self.data_path) #保存したグラーフデータの読み込み
        self.graphs = GRAPHS #グラフリストを代入
        self.labels = LABELS['label'] #ラベル辞書の値のみ代入

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


path = "./data/MyDataset/train_graphs_six_f_in_pos_50_std.dgl"
traindataset = CIFAR10TrainDataset("./data/MyDataset/train_graphs_six_f_in_pos_40.dgl")
testdataset = CIFAR10TestDataset("./data/MyDataset/test_graphs_six_f_in_pos_40.dgl")

if os.name =='posix':
    num_workers = 2
else:
    num_workers = 0
traindataloader = GraphDataLoader(traindataset,batch_size = 500,shuffle = True,num_workers = num_workers,pin_memory = True)
testdataloader = GraphDataLoader(testdataset,batch_size = 5000,shuffle = True,num_workers = num_workers,pin_memory = True)
print(f'num_wokers = {num_workers}')


#ネットワーク設定
class GCN(nn.Module):
    def __init__(self):
        super(GCN,self).__init__()
        self.conv1 = GraphConv(8,16)
        self.conv2 = GraphConv(16,32)
        self.conv3 = GraphConv(32,10)


    def forward(self,g,n_feat,e_feat = None):
        h = F.relu(self.conv1(g,n_feat,None,e_feat))
        h = F.relu(self.conv2(g,h,None,e_feat))
        h = self.conv3(g,h,None,e_feat)

        g.ndata['h'] = h

        return dgl.mean_nodes(g,'h')


model = GCN()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(),lr = 0.01)
#optimizer = optim.SGD(params=model.parameters(),lr=0.03,momentum=0.9)
epochs = 400

loss_list = []
acc_list = []
test_acc_list = []


num_correct = 0
num_tests = 0
test_num_correct = 0
test_num_tests = 0
#,batched_graph.edata['distance'].float()
BP = 0
for epoch in tqdm(range(epochs)):
    if BP != 0:
        break
    model.train()
    for batched_graph, labels in traindataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        pred = model(batched_graph, batched_graph.ndata['feat value'].float(),batched_graph.edata['distance'].float())
        loss = F.cross_entropy(pred,labels)
        if loss.item() < 0.05:
            BP = 1
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)
    loss_list.append(loss.item())
    acc_list.append(num_correct / num_tests)
    
    model.eval()
    for tbatched_graph, tlabels in testdataloader:
        tbatched_graph = tbatched_graph.to(device)
        tlabels = tlabels.to(device)
        tpred = model(tbatched_graph, tbatched_graph.ndata['feat value'],tbatched_graph.edata['distance'].float())
        test_num_correct += (tpred.argmax(1) == tlabels).sum().item()
        test_num_tests += len(tlabels)

    Tacc = test_num_correct / test_num_tests
    #print('Training accuracy:', Tacc)
    test_acc_list.append(Tacc)

num_correct = 0
num_tests = 0


with torch.no_grad():

    model.eval()
    for batched_graph, labels in traindataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        pred = model(batched_graph, batched_graph.ndata['feat value'],batched_graph.edata['distance'].float())
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)

    print('Training accuracy:', num_correct / num_tests)
    
    num_correct = 0
    num_tests = 0


    model.eval()
    for batched_graph, labels in testdataloader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)
        pred = model(batched_graph, batched_graph.ndata['feat value'].float(),batched_graph.edata['distance'].float())
        num_correct += (pred.argmax(1) == labels).sum().item()
        num_tests += len(labels)

    print('Test accuracy:', num_correct / num_tests)