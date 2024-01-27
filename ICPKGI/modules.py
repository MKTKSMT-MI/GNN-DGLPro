import dgl
from dgl.data import DGLDataset
import matplotlib.pyplot as plt
import random
import numpy as np


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
    
    
def _train_test_split(data,data_num):
    shuffle_data=random.sample(data,len(data))
    return shuffle_data[:-data_num], shuffle_data[-data_num:]


def TestAccPlot(data,dir):
    data=np.array(data)
    x=[j for j in range(data.shape[0])]
    y=data

    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(x,y)
    ax.set_title('Test accuracy')
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.set_xlim(0,data.shape[0])
    ax.set_ylim(0,1)
    fig.savefig(f'{dir}/test_acc.jpg',dpi=300)
    plt.close()


def TrainAccPlot(data,dir):
    data=np.array(data)
    x=[j for j in range(data.shape[0])]
    y=data

    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(x,y)
    ax.set_title('Train accuracy')
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.set_xlim(0,data.shape[0])
    ax.set_ylim(0,1)
    fig.savefig(f'{dir}/train_acc.jpg',dpi=300)
    plt.close()


def TrainLossPlot(data,dir):
    data=np.array(data)
    x=[j for j in range(data.shape[0])]
    y=data

    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(x,y)
    ax.set_title('Train loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_xlim(0,data.shape[0])
    fig.savefig(f'{dir}/train_loss.jpg',dpi=300)
    plt.close()


def TrainTestAccPlot(traindata,testdata,dir):
    traindata=np.array(traindata)
    testdata=np.array(testdata)
    x=[j for j in range(traindata.shape[0])]

    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(x,traindata,label='Train accuracy')
    ax.plot(x,testdata,label='Test accuracy')
    ax.legend()

    ax.set_title('Train & Test accuracy')
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    ax.set_xlim(0,traindata.shape[0])
    ax.set_ylim(0,1)
    fig.savefig(f'{dir}/train_test_acc.jpg',dpi=300)
    plt.close()