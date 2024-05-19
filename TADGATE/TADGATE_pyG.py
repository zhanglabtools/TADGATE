import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F

from .gat_conv import GATConv



class TADGATE(torch.nn.Module):
    def __init__(self, in_channels, layer1_nodes, embed_nodes):
        super(TADGATE, self).__init__()

        self.conv1 = GATConv(in_channels, layer1_nodes, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(layer1_nodes, embed_nodes, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(embed_nodes, layer1_nodes, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(layer1_nodes, in_channels, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)

    def forward(self, x, edge_index, scale_f, embed_attention = False, return_att = False):
        # x = F.dropout(x, p=0.6, training=self.training)
        if return_att == True:
            x1, att1 = self.conv1(x, edge_index, scale_f, attention=True, return_attention_weights = return_att)
        else:
            x1 = self.conv1(x, edge_index, scale_f)
        x1 = F.elu(x1)
        #x1 = F.elu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.6, training=self.training)
        # x2 = F.elu(self.conv2(x1, edge_index))
        if return_att == True and embed_attention == True:
            x2, att2 = self.conv2(x1, edge_index, scale_f, attention=embed_attention, return_attention_weights = return_att)
        else:
            x2 = self.conv2(x1, edge_index, scale_f, attention=embed_attention)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        x3 = F.elu(self.conv3(x2, edge_index, scale_f, attention=True,
                              tied_attention=self.conv1.attentions))
        x4 = self.conv4(x3, edge_index, scale_f, attention=False)
        if return_att == True and embed_attention == True:
            return x2, att1, att2, x4  # F.log_softmax(x, dim=-1)
        elif return_att == True and embed_attention == False:
            return x2, att1, x4
        elif return_att == False:
            return x2, x4