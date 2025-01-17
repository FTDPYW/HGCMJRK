import random
import os
import numpy as np
import torch
import math
from torch import nn
from efficient_kan.kan import KAN
from fastkan import FastKAN as fastKAN
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import VariLengthInputLayer, EncodeLayer, FeedForwardLayer
from torch_geometric.nn import JumpingKnowledge
device = torch.device("cuda:0")


import math



def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_torch(seed=1234)


class TransformerEncoder(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(TransformerEncoder, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.d_out = self.d_v * self.n_head * self.modal_num

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)

        # 初始化每一层的编码和前馈模块
        self.Encoder = nn.ModuleList()
        self.FeedForward = nn.ModuleList()
        self.LayerNorm = nn.ModuleList()  # 新增LayerNorm模块
        self.ResidualConnection = nn.ModuleList()  # 新增残差连接模块

        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.FeedForward.append(feedforward)

            # 每层需要一个LayerNorm和残差连接
            self.LayerNorm.append(nn.LayerNorm(self.d_k * self.n_head))
            self.ResidualConnection.append(nn.Identity())

    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, _attn = self.InputLayer(x)

        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())

        for i in range(self.n_layer):
            # 编码层的残差连接
            residual = x
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            x = self.LayerNorm[i](x + residual)  # 残差 + LayerNorm
            attn = _attn.mean(dim=1)

            # 前馈层的残差连接
            residual = x
            x = self.FeedForward[i](x)
            x = self.LayerNorm[i](x + residual)  # 残差 + LayerNorm
            attn_map.append(attn.detach().cpu().numpy())

        x = x.view(bs, -1)

        return x


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.in_features = in_ft
        self.out_features = out_ft
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):

        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)

        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class HGCN(nn.Module):
    def __init__(self, in_dim, hidden_list, dropout=0.5):
        super(HGCN, self).__init__()
        self.dropout = dropout

        # 创建 HGNN conv 层列表
        self.hgnn_layers = nn.ModuleList()
        self.hgnn_layers.append(HGNN_conv(in_dim, hidden_list[0]))  # 输入到第一个隐藏层

        # 添加其他隐藏层
        for i in range(1, len(hidden_list)):
            self.hgnn_layers.append(HGNN_conv(hidden_list[i - 1], hidden_list[i]))

        # 添加 Jumping Knowledge 机制
        self.jk = JumpingKnowledge(mode='cat')  # 'cat', 'max', or 'lstm'

    def forward(self, x, G):
        x_embeds = []

        # 通过每一层 HGNN conv
        for layer in self.hgnn_layers:
            x = layer(x, G)
            x = F.leaky_relu(x, 0.25)
            x_embeds.append(x)

        # 使用 Jumping Knowledge 机制来聚合多层特征
        x_embed = self.jk(x_embeds)

        return x_embed
#

class CL_HGCN(nn.Module):
    def __init__(self, in_size, hid_list, num_proj_hidden, alpha=0.8):
        super(CL_HGCN, self).__init__()
        self.hgcn1 = HGCN(in_size, hid_list)
        self.hgcn2 = HGCN(in_size, hid_list)

        # 根据 Jumping Knowledge 输出调整线性层的输入维度
        # 512 256 64
        # print(hid_list[-1] * len(hid_list),hid_list[-1],num_proj_hidden)
        self.fc1 = torch.nn.Linear(hid_list[-1] * len(hid_list), hid_list[-1])  # 注意调整输入维度
        self.fc1_1 = torch.nn.Linear(hid_list[-1],num_proj_hidden*2)
        self.fc1_2 = torch.nn.Linear(num_proj_hidden*2,num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_proj_hidden*2)
        self.fc3 = torch.nn.Linear(num_proj_hidden*2, num_proj_hidden )
        #tau 0.9 alpha 0.2 auc Max = 95 Avg = 94.51 aupr = 94.53
        #tau 0.5 alpha  0.8 auc = 94.52 aupr = 94.50
        self.tau = 0.5
        self.alpha = alpha


    def forward(self, x1, adj1, x2, adj2):
        z1 = self.hgcn1(x1, adj1)
        h1 = self.projection(z1)

        z2 = self.hgcn2(x2, adj2)
        h2 = self.projection(z2)

        loss = self.alpha * self.sim(h1, h2) + (1 - self.alpha) * self.sim(h2, h1)

        return z1, z2, loss

    def projection(self, z):
        z_1 = F.elu(self.fc1(z))
        z_2 = F.elu(self.fc1_1(z_1))
        z_3 = F.elu(self.fc1_2(z_2))
        z = F.elu(self.fc2(z_3))


        return self.fc3(z)

    def norm_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def sim(self, z1, z2):
        refl_sim = torch.exp(self.norm_sim(z1, z1) / self.tau)
        between_sim = torch.exp(self.norm_sim(z1, z2) / self.tau)
        loss = -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))
        loss = loss.sum(dim=-1).mean()
        return loss

class MMConv(nn.Module):
    def __init__(self, in_features, out_features,  moment=3, use_center_moment=False):
        super(MMConv, self).__init__()
        self.moment = moment
        self.use_center_moment = use_center_moment
        self.in_features = in_features

        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.w_att = Parameter(torch.FloatTensor(self.in_features * 2,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        self.w_att.data.uniform_(-stdv, stdv)
    def moment_calculation(self, x, adj_t, moment):
        mu = torch.spmm(adj_t, x)
        out_list = [mu]
        if moment > 1:
            if self.use_center_moment:# 使用中心矩时，计算二阶矩（方差）
                sigma = torch.spmm(adj_t, (x - mu).pow(2))
            else:# 不使用中心矩时，计算二阶矩（平方和）
                sigma = torch.spmm(adj_t, (x).pow(2))
            sigma[sigma == 0] = 1e-16   # 避免除零错误，将小于等于零的值设置为一个极小值
            sigma = sigma.sqrt()        # 对二阶矩取平方根，得到标准差
            out_list.append(sigma)      # 将标准差加入输出列表

            for order in range(3, moment+1):        #高阶矩
                gamma = torch.spmm(adj_t, x.pow(order))     # 计算阶数为 order 的矩
                # 处理负值情况
                mask_neg = None
                if torch.any(gamma == 0):
                    # 将等于零的值设置为一个极小值
                    gamma[gamma == 0] = 1e-16
                if torch.any(gamma < 0):
                    # 将小于零的值取相反数，并记录相应的掩码
                    mask_neg = gamma < 0
                    gamma[mask_neg] *= -1

                # 对阶数为 order 的矩取 1/order 次方根
                gamma = gamma.pow(1/order)
                if mask_neg != None:
                    gamma[mask_neg] *= -1
                # 将阶数为 order 的矩加入输出列表
                out_list.append(gamma)
        return out_list
    def attention_layer(self, moments, q):
            k_list = []
            # if self.use_norm:
            #     h_self = self.norm(h_self) # ln
            q = q.repeat(self.moment, 1) # N * m, D
            # output for each moment of 1st-neighbors
            k_list = moments
            attn_input = torch.cat([torch.cat(k_list, dim=0), q], dim=1)    #在第0维度拼接，然和与q在第1维度进行拼接
            attn_input = F.dropout(attn_input, 0.5, training=self.training)
            e = F.elu(torch.mm(attn_input, self.w_att)) # N*m, D
            attention = F.softmax(e.view(len(k_list), -1, self.out_features).transpose(0, 1), dim=1) # N, m, D  # 对注意力权重进行 softmax 归一化，得到注意力分布
            out = torch.stack(k_list, dim=1).mul(attention).sum(1) # N, D# 将每个矩按照注意力分布进行加权求和
            return out
    def forward(self, input, adj , h0 , lamda, alpha, l, beta=0.1):
        theta = math.log(lamda/l+1)
        h_agg = torch.spmm(adj, input)
        h_agg = (1-alpha)*h_agg+alpha*h0
        h_i = torch.mm(h_agg, self.weight)
        h_i = theta*h_i+(1-theta)*h_agg
        # h_moment = self.attention_layer(self.moment_calculation(input, adj, self.moment), h_i)
        h_moment = self.attention_layer(self.moment_calculation(h0, adj, self.moment), h_i)
        output = (1 - beta) * h_i + beta * h_moment
        return output


class HGCN_Attention_mechanism(nn.Module):
    def __init__(self):
        super(HGCN_Attention_mechanism, self).__init__()
        self.hiddim = 64

        self.fc_x1 = nn.Linear(in_features=2, out_features=self.hiddim)
        self.fc_x2 = nn.Linear(in_features=self.hiddim, out_features=2)
        self.sigmoidx = nn.Sigmoid()

    def forward(self, input_list):
        XM = torch.cat((input_list[0], input_list[1]), 1).t()
        XM = XM.view(1, 1 * 2, input_list[0].shape[1], -1)

        globalAvgPool_x = nn.AvgPool2d((input_list[0].shape[1], input_list[0].shape[0]), (1, 1))
        x_channel_attenttion = globalAvgPool_x(XM)

        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        x_channel_attenttion = self.fc_x1(x_channel_attenttion)
        x_channel_attenttion = torch.relu(x_channel_attenttion)
        x_channel_attenttion = self.fc_x2(x_channel_attenttion)
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1,
                                                         1)

        XM_channel_attention = x_channel_attenttion * XM
        XM_channel_attention = torch.relu(XM_channel_attention)

        return XM_channel_attention[0]





class HGCLAMIR(nn.Module):
    def __init__(self, mi_num, dis_num, hidd_list, num_proj_hidden, hyperpm):
        super(HGCLAMIR, self).__init__()

        #1
        # HGCN 模块
        self.CL_HGCN_mi = CL_HGCN(mi_num + dis_num, hidd_list, num_proj_hidden)
        #1
        self.CL_HGCN_dis = CL_HGCN(dis_num + mi_num, hidd_list, num_proj_hidden)
        #1
        # 注意力机制
        self.AM_mi = HGCN_Attention_mechanism()
        #1
        self.AM_dis = HGCN_Attention_mechanism()

        #1
        # Transformer 编码器
        self.Transformer_mi = TransformerEncoder([hidd_list[-1], hidd_list[-1]], hyperpm)
        #1
        self.Transformer_dis = TransformerEncoder([hidd_list[-1], hidd_list[-1]], hyperpm)
        #1
        # 批标准化层
        self.batch_norm_mi = nn.BatchNorm1d(200)
        #1
        self.batch_norm_dis = nn.BatchNorm1d(200)
        #1
        # MMConv 模块
        self.mmconv_mi1 = MMConv(in_features=1024, out_features=1024, moment=3, use_center_moment=True)
        #1
        self.mmconv_mi2 = MMConv(in_features=875, out_features=875, moment=3, use_center_moment=True)
       #1 auc 94.85  aupr 94.97 pre 87.89
        self.KAN = KAN([128, 64])
        self.linearmi2 = nn.Linear(875,1024)
        #1

        self.mmconv_dis1 = MMConv(in_features=1024, out_features=1024, moment=3, use_center_moment=True)
        #1
        self.mmconv_dis2 = MMConv(in_features=875, out_features=875, moment=3,
                                 use_center_moment=True)
        #1
        self.lineardis2 = nn.Linear(875, 1024)
        #1 pre 83
        # 全连接层
        self.linear_x_1 = nn.Linear(hyperpm.n_head * hyperpm.n_hidden * hyperpm.nmodal, 256)
        #1
        self.linear_x_2 = nn.Linear(256, 128)
        #1
        self.linear_x_3 = nn.Linear(128, 64)

        #1
        self.linear_y_1 = nn.Linear(hyperpm.n_head * hyperpm.n_hidden * hyperpm.nmodal, 256)
        #1
        self.linear_y_2 = nn.Linear(256, 128)
        #1
        self.linear_y_3 = nn.Linear(128, 64)

        #1



    def forward(self, concat_mi_tensor, concat_dis_tensor, G_mi_Kn, G_mi_Km, G_dis_Kn, G_dis_Km):
        # miRNA 特征
        mi_embedded = concat_mi_tensor
        mi_feature1, mi_feature2, mi_cl_loss = self.CL_HGCN_mi(mi_embedded, G_mi_Kn, mi_embedded, G_mi_Km)
        mi_feature_att = self.AM_mi([mi_feature1, mi_feature2])
        mi_feature_att1 = mi_feature_att[0].t()
        mi_feature_att2 = mi_feature_att[1].t()
        mi_concat_feature = torch.cat([mi_feature_att1, mi_feature_att2], dim=1)

        # MMConv 处理 miRNA 特征
        mi_mmconv_feature1 = self.mmconv_mi1(mi_concat_feature, G_mi_Kn, mi_concat_feature, lamda=0.01, alpha=0.11, l=1.0)
        mi_mmconv_feature2 = self.mmconv_mi2(mi_embedded, G_mi_Kn, mi_embedded, lamda=0.01, alpha=0.11, l=1.0)
        mi_mmconv_feature2 = self.linearmi2(mi_mmconv_feature2)
        mi_mmconv_feature = mi_mmconv_feature1 + mi_mmconv_feature2  # 融合处理
        # 通过 Transformer 编码 miRNA 特征（已加入残差连接）
        mi_feature = self.Transformer_mi(mi_mmconv_feature)

        # 疾病特征
        dis_embedded = concat_dis_tensor
        dis_feature1, dis_feature2, dis_cl_loss = self.CL_HGCN_dis(dis_embedded, G_dis_Kn, dis_embedded, G_dis_Km)
        dis_feature_att = self.AM_dis([dis_feature1, dis_feature2])
        dis_feature_att1 = dis_feature_att[0].t()
        dis_feature_att2 = dis_feature_att[1].t()
        dis_concat_feature = torch.cat([dis_feature_att1, dis_feature_att2], dim=1)

        # MMConv 处理疾病特征
        # print(dis_concat_feature.shape,G_dis_Kn.shape,dis_concat_feature.shape)
        dis_mmconv_feature1 = self.mmconv_dis1(dis_concat_feature, G_dis_Kn, dis_concat_feature, lamda=0.01, alpha=0.11, l=1.0)
        dis_mmconv_feature2 = self.mmconv_dis2(dis_embedded, G_dis_Kn, dis_embedded, lamda=0.01, alpha=0.11, l=1.0)
        dis_mmconv_feature2 = self.linearmi2(dis_mmconv_feature2)
        dis_mmconv_feature = dis_mmconv_feature1 + dis_mmconv_feature2  # 融合处理

        # 通过 Transformer 编码疾病特征（已加入残差连接）
        dis_feature = self.Transformer_dis(dis_mmconv_feature)

        # 批标准化
        mi_feature = self.batch_norm_mi(mi_feature)
        dis_feature = self.batch_norm_dis(dis_feature)

        # 全连接层处理
        x1 = torch.relu(self.linear_x_1(mi_feature))
        x2 = torch.relu(self.linear_x_2(x1))
        # x = torch.relu(self.linear_x_3(x2))
        x = torch.relu(self.KAN(x2))
        y1 = torch.relu(self.linear_y_1(dis_feature))
        y2 = torch.relu(self.linear_y_2(y1))
        # y = torch.relu(self.linear_y_3(y2))
        y =  torch.relu(self.KAN(y2))
        # 计算最终得分
        score = x.mm(y.t())



        return score, mi_cl_loss, dis_cl_loss