import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv,GATConv,SAGEConv


class PatchGCN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,liner=False,embedding = False):
        super(PatchGCN,self).__init__()
        self.embedding=embedding

        self.input_layer=SAGEConv(input_size,hidden_size[0], aggregator_type='mean')
        self.middle_layers=nn.ModuleList([GraphConv(hidden_size[i],hidden_size[i+1]) for i in range(len(hidden_size)-1)])
        self.output_layer=GraphConv(hidden_size[-1],output_size)
        if liner==True:
            self.liner_on = True
            self.liner_layers=nn.ModuleList([nn.Liner(hidden_size[i],hidden_size[i]) for i in range(len(hidden_size))])
        else:
            self.liner_on =False
        self.m=nn.LeakyReLU()

        self.flatt=nn.Flatten()

    
    def forward(self,g,n_feat,e_feat=None):
        n_feat=self.flatt(n_feat)
        h=self.input_layer(g,n_feat)
        h=self.m(h)
        for i,layer in enumerate(self.middle_layers):
            if self.liner_on==True:
                h=self.liner_layers[i](h)
            h=layer(g,h)
            h=self.m(h)
        g.ndata['emb'] = h
        h=self.output_layer(g,h)
        g.ndata['h'] = h

        
        if self.embedding:
            return dgl.mean_nodes(g,'h'),g
        else:
            return dgl.mean_nodes(g,'h')


class MultiPatchGCN(nn.Module):
    def __init__(self,input_size,hidden_size,object_output_size,direction_output_size,liner=False):
        super(MultiPatchGCN,self).__init__()
        self.input_layer=SAGEConv(input_size,hidden_size[0], aggregator_type='mean')
        self.middle_layers=nn.ModuleList([GraphConv(hidden_size[i],hidden_size[i+1]) for i in range(len(hidden_size)-1)])
        self.object_output_layer=GraphConv(hidden_size[-1],object_output_size) #物体用
        self.direction_output_layer=GraphConv(hidden_size[-1],direction_output_size) #方向用
        if liner==True:
            self.liner_on = True
            self.liner_layers=nn.ModuleList([nn.Liner(hidden_size[i],hidden_size[i]) for i in range(len(hidden_size))])
        else:
            self.liner_on =False
        self.m=nn.LeakyReLU()

        self.flatt=nn.Flatten()

    
    def forward(self,g,n_feat,e_feat=None):
        #入力層
        n_feat=self.flatt(n_feat)
        h=self.input_layer(g,n_feat)
        h=self.m(h)

        #中間層
        for i,layer in enumerate(self.middle_layers):
            if self.liner_on==True:
                h=self.liner_layers[i](h)
            h=layer(g,h)
            h=self.m(h)

        #埋め込み出力
        g.ndata['emb'] = h

        #出力層-物体
        o=self.object_output_layer(g,h)
        g.ndata['o'] = o

        #出力層-方向
        d=self.direction_output_layer(g,h)
        g.ndata['d']=d

        
        return dgl.mean_nodes(g,'o'),dgl.mean_nodes(g,'d'),g
    


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
    

class EmbeddingNetwork(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(EmbeddingNetwork, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
        self.conv3 = SAGEConv(hidden_feats, out_feats, aggregator_type='mean')
        '''self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, hidden_feats)'''
        self.flatt=nn.Flatten()

    def forward(self, g, features):
        '''x=self.flatt(features)
        x = torch.relu(self.conv1(g, x))
        x = self.conv2(g, x)
        #x = self.conv3(g,x)'''
        
        x = self.flatt(features)
        x = torch.relu(self.conv1(g,x))
        x = torch.relu(self.conv2(g,x))
        x = self.conv3(g,x)

        g.ndata['h'] = x
        return g