import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter_mean,scatter_add
from torch_geometric.nn import TransformerConv
from data import *
from torch_geometric.nn import global_mean_pool,global_max_pool,global_add_pool
        
        
class GNNLayer(nn.Module):
    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(in_channels=num_hidden, out_channels=int(num_hidden / num_heads), heads=num_heads, dropout = dropout, edge_dim = num_hidden, root_weight=False)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        # update edge
        h_E = self.edge_update(h_V, edge_index, h_E)

        # context node update
        h_V = self.context(h_V, batch_id)

        return h_V, h_E


class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3*num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E


class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

    def forward(self, h_V, batch_id):
        # c_V = scatter_add(h_V, batch_id, dim=0)
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V


class Graph_encoder(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim,
                 seq_in=False, num_layers=4, drop_rate=0.2):
        super(Graph_encoder, self).__init__()

        self.seq_in = seq_in
        if self.seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim += 20
        
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)
        
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.layers = nn.ModuleList(
                GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4)
            for _ in range(num_layers))


    def forward(self, h_V, edge_index, h_E, seq, batch_id):
        if self.seq_in and seq is not None:
            seq = self.W_s(seq)
            h_V = torch.cat([h_V, seq], dim=-1)

        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)
        
        return h_V

class Attention(nn.Module):
    def __init__(self, input_dim, dense_dim, n_heads):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.n_heads = n_heads
        self.fc1 = nn.Linear(self.input_dim, self.dense_dim)
        self.fc2 = nn.Linear(self.dense_dim, self.n_heads)

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = torch.softmax(input_2d, dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, input):  				# input.shape = (1, seq_len, input_dim)
        x = torch.tanh(self.fc1(input))  	# x.shape = (1, seq_len, dense_dim)
        x = self.fc2(x)  					# x.shape = (1, seq_len, attention_hops)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)  		# attention.shape = (1, attention_hops, seq_len)
        return attention

class GraphEC(nn.Module): 
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, num_layers, dropout, augment_eps, device):
        super(GraphEC, self).__init__()
        self.augment_eps = augment_eps
        self.device = device
        self.hidden_dim = hidden_dim
        self.node_input_dim = node_input_dim
        self.Graph_encoder = Graph_encoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim, hidden_dim=hidden_dim, seq_in=False, num_layers=num_layers, drop_rate=dropout)

        self.attention = Attention(hidden_dim,dense_dim=16,n_heads=4)
        
        self.input_block = nn.Sequential(
                                         nn.LayerNorm(1024+9, eps=1e-6)
                                        ,nn.Linear(1024+9, hidden_dim)
                                        ,nn.LeakyReLU()
                                        )
        attention_heads = 8
        self.output_block = nn.Sequential(
                                         nn.Linear((attention_heads+1)*hidden_dim, (attention_heads+1)*hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm((attention_heads+1)*hidden_dim, eps=1e-6)
                                         ,nn.Dropout(dropout)
                                         ,nn.Linear((attention_heads+1)*hidden_dim,5106)
                                         )
        num_emb_layers = 2
        self.hidden_block = []
        for i in range(num_emb_layers - 1):
            self.hidden_block.extend([
                                      nn.LayerNorm(hidden_dim, eps=1e-6)
                                     ,nn.Dropout(dropout)
                                     ,nn.Linear(hidden_dim, hidden_dim)
                                     ,nn.LeakyReLU()
                                     ])
            if i == num_emb_layers - 2:
                self.hidden_block.extend([nn.LayerNorm(hidden_dim, eps=1e-6)])

        self.hidden_block = nn.Sequential(*self.hidden_block)
        
        # Attention pooling layer
        self.ATFC = nn.Sequential(
                                  nn.Linear(hidden_dim, 64)
                                 ,nn.LeakyReLU()
                                 ,nn.LayerNorm(64, eps=1e-6)
                                 ,nn.Linear(64, attention_heads) # num_heads
                                 )
        self.weight = 0.2
        self.add_module("FC_1", nn.Linear(hidden_dim, hidden_dim, bias=True))
        self.add_module("FC_2", nn.Linear(hidden_dim, 5106, bias=True))

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  
        
    def padding_ver1(self, x, batch_id, feature_dim):
        batch_size = max(batch_id) + 1
        max_len= max(torch.unique(batch_id,return_counts=True)[1])
        batch_data = torch.zeros([batch_size,max_len,feature_dim])
        mask = torch.zeros([batch_size,max_len])
        len_0 = 0
        len_1 = 0
        for i in range(batch_size):
            len_1 = len_0 + torch.unique(batch_id,return_counts=True)[1][i]
            batch_data[i][:torch.unique(batch_id,return_counts=True)[1][i]] = x[len_0:len_1]
            mask[i][:torch.unique(batch_id,return_counts=True)[1][i]] = 1
            len_0 += torch.unique(batch_id,return_counts=True)[1][i]
        return batch_data, mask

    def forward(self, X, h_V, edge_index, seq, batch_id,batch_data, mask_data, batch_activate_site):
        # Data augmentation
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
            h_V = h_V + self.augment_eps * torch.randn_like(h_V)

        h_V_baseline, _ = batch_data, mask_data
        h_V_baseline = h_V_baseline.to(self.device)
        h_V_baseline = self.input_block(h_V_baseline)
        h_V_baseline = self.hidden_block(h_V_baseline)

        h_V_geo, h_E = get_geo_feat(X, edge_index)
        try:
            h_V = torch.cat([h_V, h_V_geo], dim=-1)  
        except:
            print(h_V.size()) 
            print(h_V_geo.size())  
            print(seq)
            h_V_geo = torch.ones([h_V.shape[0],184]).to(self.device)
            h_V = torch.cat([h_V, h_V_geo], dim=-1)


        h_V = h_V.to(self.device)
        h_V = self.Graph_encoder(h_V, edge_index, h_E, seq, batch_id) # [num_residue, hidden_dim]
        h_V_stru, mask_baseline = self.padding_ver1(h_V.cpu(), batch_id.cpu(), h_V.shape[1])
        h_V_stru = h_V_stru.to(self.device)
        mask_baseline = mask_baseline.to(self.device)
        
        h_V_baseline = self.weight*h_V_baseline + (1-self.weight)*h_V_stru

        # Attention pooling
        att = self.ATFC(h_V_baseline)    # [B, L, hid] -> [B, L, att_heads]
        att = att.masked_fill(mask_baseline[:, :, None] == 0, -1e9)
        att = F.softmax(att, dim=1) # [B, L, att_heads]
        
        active_pool = batch_activate_site.transpose(1,2)@h_V_baseline
        
        att = att.transpose(1,2)   # [B, L, att_heads] -> [B, att_heads, L]
        h_V_baseline = att@h_V_baseline    # [B, att_heads, hid]
        
        h_V_baseline = torch.cat((h_V_baseline,active_pool),1)
        
        h_V_baseline = torch.flatten(h_V_baseline, start_dim=1) # [B, att_heads,hid] -> [B, att_heads*hid]

        h_V_baseline = self.output_block(h_V_baseline) # [B, d_label]

        return h_V_baseline.float()
