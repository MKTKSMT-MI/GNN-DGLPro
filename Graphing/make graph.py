#必要なパッケージをインポート
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import urllib.request
import dgl
from dgl.data import DGLDataset
import os
import itertools
import time
import sys


#CIFAR-10の読み込み
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root = '../data', train = True, download = True, transform = transform)
testset = torchvision.datasets.CIFAR10(root='../data',train=False,download=True,transform = transform)

#畳み込み結果を取得するためのネットワーククラス
class Conv_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        return x


class Dence_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Nested_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = Conv_layer()
        self.Dence_net = Dence_layer()

    def forward(self,x):
        x = self.conv_net(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.Dence_net(x)

        return x


#処理を高速化するためにGPUを使用するための設定。モデルをGPUへ
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Nested_net()
model.to(device)

#学習済みモデルをロード
path = '../models/Nest_model.pth'
model.load_state_dict(torch.load(path))

def get_distance(x,y): #距離計算用関数
    hdist = ((x[0] - y[0])**2 + (x[1] - y[1])**2)
    #hdist = torch.tensor(hdist)
    hdist = hdist.clone().detach()
    dist = torch.sqrt(hdist)

    return dist


def return_two_list(node_num):
    taikaku = torch.full((node_num,node_num),fill_value=1.)
    for i in range(node_num):
        taikaku[i][i] = 0.
    src_ids = []
    dst_ids = []
    for i in range(node_num):
        for j in range(i,node_num):
            if taikaku[i][j] != 0:
                src_ids.append(i)
                dst_ids.append(j)
                src_ids.append(j)
                dst_ids.append(i)
    tensor_src = torch.tensor(src_ids)
    tensor_dst = torch.tensor(dst_ids)
    return tensor_src,tensor_dst

    
def zscore(x,axis = 0):
    stddev, mean = torch.std_mean(x,dim=axis,unbiased=True)
    result = (x - mean)/stddev
    return result


#設定
pool = nn.MaxPool2d(2,2)
train_data_number = 2000 #0~5000 or full
num_node = 40
graph_type = 1 #1(n:feat value, e:distance) or 2(n:distance, e:none)
mode = None
save1_file_name = f'{mode}_6feat_dist_{num_node}.dgl'
save2_file_name = f'{mode}_dist_{num_node}.dgl'


graphs = []
labels = []
class_counter = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0} #カウント用辞書
train_class_counter = {0:train_data_number,1:train_data_number,2:train_data_number,3:train_data_number,4:train_data_number,
                       5:train_data_number,6:train_data_number,7:train_data_number,8:train_data_number,9:train_data_number}
src,dst = return_two_list(num_node)

#train start
mode = 'train'
if type(train_data_number) is int:
    check = True
    flag = False
else:
    check = False
for image,label in tqdm(trainset):
    if check == True:
        if class_counter == train_class_counter:
            break
        if class_counter[label] == train_data_number:
            continue
        else:
            class_counter[label] += 1
    image = image.to(device)
    image = image.unsqueeze(0)
    features_maps = F.relu(model.conv_net.conv1(image))#BCSS
    image_size = features_maps.shape[2]
    #print(features_maps.shape)
    features_maps = features_maps.permute(1,2,3,0)#CSSB
    #print(features_maps.shape)
    synthetic_map = 0
    for i in features_maps:#SSC
        synthetic_map += i
    features_maps = torch.squeeze(features_maps)
    #print(features_maps.shape)
    synthetic_map = torch.squeeze(synthetic_map)#SS
    zero2one_map = synthetic_map/torch.max(synthetic_map)
    onedim = zero2one_map.reshape(image_size * image_size)
    #print(onedim.shape)
    sort_onedim,sort_index = torch.sort(onedim)
    #print(sort_onedim[-5:],sort_index[-5:])
    ori_index = torch.empty(num_node,2,dtype=torch.float32,device=device)
    for i in range(num_node):
        x = sort_index[-num_node:][i] // image_size
        y = sort_index[-num_node:][i] - x * image_size
        ori_index[i][0] = x
        ori_index[i][1] = y
    #print(ori_index)
    distance = []
    for i in range(num_node-1):
        for j in range(i+1,num_node):
            distance.append(get_distance(ori_index[i],ori_index[j]))
    distance = torch.tensor(distance,device=device)
    #print(distance)
    if graph_type == 1:
        node_data = torch.empty([num_node,6],device=device)
        for i in range(num_node):
            for j in range(6):
                x = int(ori_index[i][0].item())
                y = int(ori_index[i][1].item())
                node_data[i][j] = features_maps[j][x][y]
        #print(node_data)
        edge_data = torch.empty([num_node * (num_node - 1),1],device=device)
        for i,dis in enumerate(distance):
            edge_data[i*2] = dis
            edge_data[i*2+1] = dis
        #print(edge_data)

        g = dgl.graph((src,dst),num_nodes=num_node,device=device)
        g.ndata['feat value'] = node_data
        g.edata['distance'] = edge_data
        graphs.append(g)
        labels.append(label)
    elif graph_type == 2:
        node_data = torch.zeros([num_node,num_node],device=device)
        d = 0
        for i in range(num_node - 1):
            node_data[i][i+1:num_node] = distance[d:d + (num_node -1)+i*-1]
            d = d + (num_node -1)+i*-1
        node_data = node_data + torch.transpose(node_data,0,1)
        #print(node_data)

        g = dgl.graph((src,dst),num_nodes=num_node,device=device)
        g.ndata['feat value'] = node_data
        graphs.append(g)
        labels.append(label)
    else:
        print('graph_typeの指定が不正。終了します。')
        sys.exit()
print(len(graphs))
print(len(labels))
print(class_counter)
print(f'ノード数: {num_node}')
output_labels = {'label':torch.tensor(labels)}
if graph_type == 1:
    path = f'../data/NewMyData/{mode}_6feat_dist_{num_node}.dgl'
elif graph_type == 2:
    path = f'../data/NewMyData/{mode}_dist_{num_node}.dgl'
dgl.save_graphs(path,g_list=graphs,labels=output_labels)


graphs = []
labels = []
class_counter = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0} #カウント用辞書
src,dst = return_two_list(num_node)

#train start
mode = 'test'
for image,label in tqdm(testset):
    class_counter[label] += 1
    image = image.to(device)
    image = image.unsqueeze(0)
    features_maps = F.relu(model.conv_net.conv1(image))#BCSS
    image_size = features_maps.shape[2]
    #print(features_maps.shape)
    features_maps = features_maps.permute(1,2,3,0)#CSSB

    synthetic_map = 0
    for i in features_maps:#SSC
        synthetic_map += i
    features_maps = torch.squeeze(features_maps)
    synthetic_map = torch.squeeze(synthetic_map)#SS
    zero2one_map = synthetic_map/torch.max(synthetic_map)
    onedim = zero2one_map.reshape(image_size * image_size)
    #print(onedim.shape)
    sort_onedim,sort_index = torch.sort(onedim)
    #print(sort_onedim[-5:],sort_index[-5:])
    ori_index = torch.empty(num_node,2,dtype=torch.float32,device=device)
    for i in range(num_node):
        x = sort_index[-num_node:][i] // image_size
        y = sort_index[-num_node:][i] - x * image_size
        ori_index[i][0] = x
        ori_index[i][1] = y

    distance = []
    for i in range(num_node-1):
        for j in range(i+1,num_node):
            distance.append(get_distance(ori_index[i],ori_index[j]))
    distance = torch.tensor(distance,device=device)
 
    if graph_type == 1:
        node_data = torch.empty([num_node,6],device=device)
        for i in range(num_node):
            for j in range(6):
                x = int(ori_index[i][0].item())
                y = int(ori_index[i][1].item())
                node_data[i][j] = features_maps[j][x][y]

        edge_data = torch.empty([num_node * (num_node - 1),1],device=device)
        for i,dis in enumerate(distance):
            edge_data[i*2] = dis
            edge_data[i*2+1] = dis

        g = dgl.graph((src,dst),num_nodes=num_node,device=device)
        g.ndata['feat value'] = node_data
        g.edata['distance'] = edge_data
        graphs.append(g)
        labels.append(label)

    elif graph_type == 2:
        node_data = torch.zeros([num_node,num_node],device=device)
        d = 0
        for i in range(num_node - 1):
            node_data[i][i+1:num_node] = distance[d:d + (num_node -1)+i*-1]
            d = d + (num_node -1)+i*-1
        node_data = node_data + torch.transpose(node_data,0,1)
        
        g = dgl.graph((src,dst),num_nodes=num_node,device=device)
        g.ndata['feat value'] = node_data
        graphs.append(g)
        labels.append(label)

    else:
        print('graph_typeの指定が不正。終了します。')
        sys.exit()

print(len(graphs))
print(len(labels))
print(class_counter)
print(f'ノード数: {num_node}')
output_labels = {'label':torch.tensor(labels)}
if graph_type == 1:
    path = f'../data/NewMyData/{mode}_6feat_dist_{num_node}.dgl'
elif graph_type == 2:
    path = f'../data/NewMyData/{mode}_dist_{num_node}.dgl'
dgl.save_graphs(path,g_list=graphs,labels=output_labels)