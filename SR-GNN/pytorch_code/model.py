#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        # 有关Parameter函数的解释：首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，
        # 所以经过类型转换这个self.XX变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。——————https://www.jianshu.com/p/d8b77cc02410
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        # 有关nn.Linear的解释：torch.nn.Linear(in_features, out_features, bias=True)，对输入数据做线性变换：y=Ax+b
        # nn.Linear内部调用了F.linear，并自动地对参数进行了初始化。
        # 形状：输入: (N,in_features)  输出： (N,out_features)
        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        # A-->实际上是该batch数据图矩阵的列表  eg:(100,5?,10?(即5?X2))
        # hidden--> eg(100-batch_size,5?,100-embeding_size)，hidden相当于v_t-1 ？？？
        # 后面所有的5?代表这个维的长度是该batch唯一最大类别长度(类别数目不足该长度的session补0)，根据不同batch会变化
        # 有关matmul的解释：矩阵相乘，多维会广播相乘。
        # 取出A的入度矩阵部分
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah # input_in：(100,5?,100)
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah  # input_out(100,5?,100)
        # 在最后一个维度上将两个矩阵拼接起来。即论文As一样的形状
        inputs = torch.cat([input_in, input_out], 2)  # inputs-->(100,5?,200)
        # 关于functional.linear(input, weight, bias=None)的解释：y= xA^T + b 应用线性变换，
        # input:(N，*，in_features),Output: (N,*,out_features),*代表此维度的任意数值
        gi = F.linear(inputs, self.w_ih, self.b_ih)  # gi(100,5?,300)
        gh = F.linear(hidden, self.w_hh, self.b_hh)  # gh(100,5?,300)
        i_r, i_i, i_n = gi.chunk(3, 2)  # 3是划分的块的数量，2是划分的维度
        h_r, h_i, h_n = gh.chunk(3, 2)  # 将(100,5?,300)沿着第三个维度划分为三个tensor，各自形状为(100,5?,100)
        resetgate = torch.sigmoid(i_r + h_r)  # 公式(3)，resetgate(100,5?,100)
        inputgate = torch.sigmoid(i_i + h_i)  # 公式(2)，inputgate(100,5?,100)
        newgate = torch.tanh(i_n + resetgate * h_n)  # 公式(4)，newgate(100,5?,100)
        hy = newgate + inputgate * (hidden - newgate)  # 公式(5)，hy(100,5?,100)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class SessionGraph(Module):
    def __init__(self, opt, n_node):  # n_node 存储每个输入序列u_input中item类别数，即去除重复点击项的剩余个数。（包括0）
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)  #　ｎ_node表示词典大小，hidden_size表示嵌入的维度。
        self.gnn = GNN(self.hidden_size, step=opt.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()  # 初始化权重参数

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)  # stdv=0.1
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)  # 将参数初始化到-0.1-0.1之间

    def compute_scores(self, hidden, mask):
        # hidden-->(100,16?,100) 其中16?代表该样本所有数据最长会话的长度(不同数据集会不同)，单个样本其余部分补了0
        # mask-->(100,16?) 有序列的位置是[1],没有动作序列的位置是[0]
        # ht是最后一个点击。用来表示短期兴趣
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size   Wv_n
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size Wv_i
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # 公式(6)的α：batch_size x seq_length x 1 (100*16*1)
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)  # '*'为对应位置相乘。公式(6)的s:100*100
        if not self.nonhybrid:  # nonhybrid表示仅使用全局偏好
            a = self.linear_transform(torch.cat([a, ht], 1))  # 公式(7)
        b = self.embedding.weight[1:]  # n_nodes x latent_size  选取每个节点的embedding向量。去除第一个0，因为0是填充的
        scores = torch.matmul(a, b.transpose(1, 0))  # 公式(8)
        return scores  # scores：100*nodes

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):  # 模型model(SessionGraph)；i是每个batch的session的索引，eg[0,1,...,99]；data是Data的实例对象
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    #nn.Module会自动调用forward函数，所以不需要写成model.forward
    hidden = model(items, A)  # 调用SessionGraph的forward函数，返回维度（100*5？*100）
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])  # 按照session进行打包
    return targets, model.compute_scores(seq_hidden, mask)  # 返回target和对应的预测分数


def train_test(model, train_data, test_data):  # 这里的model是SessionGraph的实例对象

    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    # slices 是所有session按照batchsize分完批后的每批的序号[[0,...99],[100,199],...[]]
    slices = train_data.generate_batch(model.batch_size)  # train_data是在main定义的Data()类的对象，可以使用utils中的方法
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)  # targets - 1保持和填充0后的数据保持一致
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:  # len(slices)就是batch的个数，其中batchsize=100
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    model.scheduler.step()  # scheduler.step()用来更新优化器的学习率，按照epoch单位进行更换
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores = forward(model, i, test_data)  # targets(batch*1),scores(batch*100)
        sub_scores = scores.topk(20)[1]  # 求出每一行的top20的值。sub_scores(batch*20)
        # 关于detach，tensor.data和tensor.detach()都是从原有tensor中分离出的数据。
        # tensor.detach()能被autograd()追踪求导。x.data则不可以
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()  # 变为array类型
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))  # 判断target的元素是否在score中出现过。
            if len(np.where(score == target - 1)[0]) == 0:  # 判读score和target对应位置是否相等
                mrr.append(0)  # 若20个里面没有一个排名正确的，则MRR的值为0
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))  # MRR:排名的倒数
    hit = np.mean(hit) * 100  # 计算均值
    mrr = np.mean(mrr) * 100  # 排名倒数的均值
    return hit, mrr
