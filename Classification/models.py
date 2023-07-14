import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class DynamicGCN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(DynamicGCN,self).__init__()
        self.input_layer=GraphConv(input_size,hidden_size[0])
        self.middle_layers=nn.ModuleList([GraphConv(hidden_size[i],hidden_size[i+1]) for i in range(len(hidden_size)-1)])
        self.output_layer=GraphConv(hidden_size[-1],output_size)

        self.flatt=nn.Flatten()

    
    def forward(self,g,n_feat,e_feat=None):
        n_feat=self.flatt(n_feat)
        h=self.input_layer(g,n_feat,None,e_feat).clamp(0)
        for layer in self.middle_layers:
            h=layer(g,h).clamp(0)
        h=self.output_layer(g,h).clamp(0)
        g.ndata['h'] = h

        return dgl.mean_nodes(g,'h')