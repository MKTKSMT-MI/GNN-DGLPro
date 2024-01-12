import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv,GATConv,SAGEConv


class PatchGCN(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super(PatchGCN,self).__init__()
        self.input_layer=GraphConv(input_size,hidden_size[0])
        self.middle_layers=nn.ModuleList([GraphConv(hidden_size[i],hidden_size[i+1]) for i in range(len(hidden_size)-1)])
        self.output_layer=GraphConv(hidden_size[-1],output_size)
        
        self.m=nn.LeakyReLU()
        self.flatt=nn.Flatten()

    
    def forward(self,g,n_feat):
        n_feat=self.flatt(n_feat)
        h=self.input_layer(g,n_feat)
        h=self.m(h)
        for i,layer in enumerate(self.middle_layers):
            h=layer(g,h)
            h=self.m(h)
        h=self.output_layer(g,h)
        h=self.m(h)
        g.ndata['h'] = h

        return g
    


class PatchSAGE(nn.Module):
    def __init__(self, input_size,hidden_size,output_size):
        super(PatchGCN,self).__init__()
        self.input_layer=SAGEConv(input_size,hidden_size[0], aggregator_type='mean')
        self.middle_layers=nn.ModuleList([SAGEConv(hidden_size[i],hidden_size[i+1], aggregator_type='mean') for i in range(len(hidden_size)-1)])
        self.output_layer=SAGEConv(hidden_size[-1],output_size, aggregator_type='mean')
        
        self.m=nn.LeakyReLU()
        self.flatt=nn.Flatten()

    
    def forward(self,g,n_feat):
        n_feat=self.flatt(n_feat)
        h=self.input_layer(g,n_feat)
        h=self.m(h)
        for i,layer in enumerate(self.middle_layers):
            h=layer(g,h)
            h=self.m(h)
        h=self.output_layer(g,h)
        h=self.m(h)
        g.ndata['h'] = h

        return g
    

class PatchGAT(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_head):
        super(PatchGAT,self).__init__()

        self.input_layer=GATConv(input_size, hidden_size[0], num_head, feat_drop=0.2) #inputsize => hiddensize0 * numhead
        self.input_Dence_layer=nn.Linear(hidden_size[0]*num_head,hidden_size[0]) #hiddensize0 * numhead => hiddensize0

        self.middle_layers=nn.ModuleList([GATConv(hidden_size[i], hidden_size[i+1], num_head, feat_drop=0.4) for i in range(len(hidden_size)-1)])
        self.catDence_layers=nn.ModuleList([nn.Linear(hidden_size[i]*num_head,hidden_size[i]) for i in range(1,len(hidden_size))])

        self.output_layer=GATConv(hidden_size[-1], output_size, num_head, feat_drop=0.4)
        
        self.m=nn.LeakyReLU()

        self.flatt=nn.Flatten()

    def forward(self,g,n_feat,e_feat=None):
        n_feat=self.flatt(n_feat)
        h=self.input_layer(g,n_feat)
        h=self.m(h)
        h=torch.flatten(h,-2)
        h=self.input_Dence_layer(h)
        h=self.m(h)
        for i,layer in enumerate(self.middle_layers):
            h=layer(g,h)
            h=self.m(h)
            h=torch.flatten(h,-2)
            h=self.catDence_layers[i](h)
            h=self.m(h)
        h=self.output_layer(g,h)
        h=self.m(h)
        h=torch.mean(h,-2)
        h=self.m(h)
        g.ndata['h'] = h
        return g