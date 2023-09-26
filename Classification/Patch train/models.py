import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv,GATConv,GATv2Conv

class PatchGCN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,linear=False):
        super(PatchGCN,self).__init__()
        self.linear_on = linear
        self.input_layer=GraphConv(input_size,hidden_size[0])
        self.middle_layers=nn.ModuleList([GraphConv(hidden_size[i],hidden_size[i+1]) for i in range(len(hidden_size)-1)])
        self.output_layer=GraphConv(hidden_size[-1],output_size)
        if self.linear_on:
            self.linear_layers=nn.ModuleList([nn.Linear(hidden_size[i],hidden_size[i]) for i in range(len(hidden_size))])
        self.m=nn.LeakyReLU()

        self.flatt=nn.Flatten()

    
    def forward(self,g,n_feat,e_feat=None):
        n_feat=self.flatt(n_feat)
        h=self.input_layer(g,n_feat,None,e_feat)
        h=self.m(h)
        for i,layer in enumerate(self.middle_layers):
            if self.linear_on:
                h=self.linear_layers[i](h)
            h=layer(g,h)
            h=self.m(h)
        h=self.output_layer(g,h)
        h=self.m(h)
        g.ndata['h'] = h

        return dgl.mean_nodes(g,'h')
    

class PatchGAT(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_head,linear=False):
        super(PatchGAT,self).__init__()
        self.linear_on=linear
        self.sum_k=len(hidden_size)+1
        self.input_layer=GATConv(input_size, hidden_size[0], num_head, feat_drop=0.4,residual=True)
        self.middle_layers=nn.ModuleList([GATConv(hidden_size[i], hidden_size[i+1], num_head, feat_drop=0.4,residual=True) for i in range(len(hidden_size)-1)])
        self.output_layer=GATConv(hidden_size[-1], output_size, num_head, feat_drop=0.4,residual=True)
        self.fc=nn.Linear((num_head**self.sum_k)*output_size,output_size)
        self.m=nn.LeakyReLU()

        self.flatt=nn.Flatten()

    def forward(self,g,n_feat,e_feat=None):
        n_feat=self.flatt(n_feat)
        h=self.input_layer(g,n_feat)
        h=self.m(h)
        for i,layer in enumerate(self.middle_layers):
            h=layer(g,h)
            h=self.m(h)
        h=self.output_layer(g,h)
        h=self.m(h)
        h=self.flatt(h)
        h=self.fc(h)
        g.ndata['h'] = h
        
        return dgl.mean_nodes(g,'h')
    

class PatchGATv2(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_head,linear=False):
        super(PatchGATv2,self).__init__()
        self.linear_on=linear
        self.sum_k=len(hidden_size)+1
        self.input_layer=GATv2Conv(input_size, hidden_size[0], num_head, feat_drop=0.5)
        self.middle_layers=nn.ModuleList([GATv2Conv(hidden_size[i], hidden_size[i+1], num_head, feat_drop=0.7) for i in range(len(hidden_size)-1)])
        self.output_layer=GATv2Conv(hidden_size[-1], output_size, num_head, feat_drop=0.6)
        self.fc=nn.Linear((num_head**self.sum_k)*output_size,output_size)
        self.m=nn.LeakyReLU()

        self.flatt=nn.Flatten()

    def forward(self,g,n_feat,e_feat=None):
        n_feat=self.flatt(n_feat)
        h=self.input_layer(g,n_feat)
        #print(f'one h shape: {h.shape}')
        h=h.mean(1)
        h=self.m(h)
        for i,layer in enumerate(self.middle_layers):
            h=layer(g,h)
            h=h.mean(1)
            h=self.m(h)
        h=self.output_layer(g,h)
        h=h.mean(1)
        h=self.m(h)
        g.ndata['h'] = h
        
        return dgl.mean_nodes(g,'h')