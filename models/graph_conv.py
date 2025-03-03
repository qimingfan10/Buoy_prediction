import dgl
import torch.nn as nn

class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'h'))
            h = g.ndata['h']
            return self.linear(h) 