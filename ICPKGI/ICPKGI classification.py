import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dgl.nn import GraphConv,SAGEConv
import dgl
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import seaborn as sns
import random
from tqdm import tqdm
import sklearn
from sklearn.model_selection import train_test_split
import os
import yaml
import time
import datetime
from modules import ICPKGIDataset
from models import PatchGCN



device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
object_name = 'airplane'  #car bus airplane
setting_file = "config2.yaml"

#データ読み込み
dataset=ICPKGIDataset(f'GNN-DGLPro/data/ICPKGI/8patch_gray_{object_name}.dgl')

#各クラスから均等に10個ずつテスト用として抜き出しtrainデータセットとtestデータセットを作成
labels=[i.item() for _,i in dataset]
traindataset, testdataset, trainlabels, testlabels=train_test_split(dataset,labels,test_size=0.2,shuffle=True,stratify=labels)

#データローダー作成
traindataloader=GraphDataLoader(traindataset,batch_size=16,shuffle=True,num_workers = 0,pin_memory = True)
testdataloader=GraphDataLoader(testdataset,batch_size=10,shuffle=True,num_workers = 0,pin_memory = True)

#全学習を通して変わらない事前情報
#トレーニングデータのそれぞれのクラス数
train_label_num=[0]*5
for i in trainlabels:
    train_label_num[i]+=1

#設定ファイル読み込み
with open(f'GNN-DGLPro/ICPKGI/configs/{setting_file}','r') as f:
    config = yaml.safe_load(f)

'''#モデルの初期化
model=PatchGCN(1024,[8],5,embedding=True)
model.to(device)'''

#パラメータ設定
lr = 0.0001
epochs = 20
get_embedding=True
cos=nn.CosineSimilarity(-1)

print(f'object name: {object_name}')
for model_name, model_config in config.items():
    #時間計測
    start=time.time()
    #結果を保存するディレクトリを作成
    #Classification/save
    #save_dir=f'../Classification/save/{data_path[data_number]}/config1.yaml/{model_name}'
    #save_dir=f'../../Classification/save/embedding/single class/{object_name}/{model_name}'
    save_dir=f'GNN-DGLPro/ICPKGI/save/embedding/single class/{object_name}/{model_name}'
    os.makedirs(save_dir,exist_ok=True)

    #モデルの初期化
    #model=PatchGCN(model_config['input_size'],model_config['hidden_size'],model_config['output_size'])
    model=PatchGCN(1024,[8],5,embedding=get_embedding)
    model.to(device)
    lossF=nn.CrossEntropyLoss()
    optimizer=optim.AdamW(model.parameters(),lr=lr)

    #情報保存用の変数の初期化
    #トレーニング用
    num_correct=0
    num_tests=0
    loss_correct=0
    train_loss_list = []
    train_acc_list = []
    #テスト用
    test_num_correct = 0
    test_num_tests = 0
    best_acc=0
    test_acc_list = []
    test_emb_acc_list=[]
    print('epochスタート')
    for epoch in tqdm(range(epochs)):
        train_emb_graphs=[]
        train_emb_labels=[]
        model.train()
        for i,(batched_graph,labels) in enumerate(traindataloader):
            train_emb_labels.extend(labels.tolist())
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            
            pred,emb = model(batched_graph,batched_graph.ndata['f'])
            train_emb_graphs.extend(dgl.unbatch(emb))
            loss=lossF(pred,labels)
            loss_correct+=loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_correct += (pred.argmax(1) == labels).sum().item()
            num_tests += len(labels)
        train_loss_list.append(loss_correct / (i+1))
        train_acc_list.append(num_correct / num_tests)
        #カウントリセット
        num_correct=num_tests=loss_correct=0

        #類似度分類
        direction_graphs=torch.zeros(5,64,8) #中間層の出力のクラス特徴を保存するリスト
        for emb_graph,emb_label in zip(train_emb_graphs,train_emb_labels):
            #中間層の出力をラベル数で割ってクラスインデックスに加算する
            direction_graphs[emb_label]+=((emb_graph.ndata['emb'])/train_label_num[emb_label]).to('cpu')
        #print(f'direction_graphs shape:{direction_graphs.shape}')
        #print(f'direction_labels:{train_label_num}')
        

        test_emb_graphs=[]
        test_emb_labels=[]
        #テスト
        model.eval()
        for tbatched_graph, tlabels in testdataloader:
            test_emb_labels.extend(tlabels.tolist())
            tbatched_graph = tbatched_graph.to(device)
            tlabels = tlabels.to(device)
            tpred,temb = model(tbatched_graph, tbatched_graph.ndata['f'])
            test_emb_graphs.extend(dgl.unbatch(temb))

            tpred = F.softmax(tpred,dim=1)
            test_num_correct += (tpred.argmax(1) == tlabels).sum().item()
            test_num_tests += len(tlabels)

        test_acc_list.append(test_num_correct/test_num_tests)
        if best_acc < test_num_correct/test_num_tests:
            best_acc = test_num_correct/test_num_tests
            best_weight = model
        #カウントリセット
        test_num_correct=test_num_tests=0
        #類似度計算
        stack_test_emb = torch.stack([g.ndata['emb'].to('cpu') for g in test_emb_graphs],dim=0).unsqueeze(1)
        #print(f'stack_test_emb shape:{stack_test_emb.shape}')
        emb_pred = torch.sum(cos(direction_graphs,stack_test_emb),dim=-1)
        test_emb_correct=(emb_pred.argmax(1)==torch.tensor(test_emb_labels)).sum().item()
        test_emb_acc = test_emb_correct/len(test_emb_labels)
        #print(f'test embedding acc:{test_emb_acc*100}%')
        test_emb_acc_list.append(test_emb_acc)
    print(f'{epochs} acc : {test_emb_acc*100}')
    plt.plot(range((epochs)),test_emb_acc_list)
    plt.show()