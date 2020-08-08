#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import networkx as nx
import numpy as np

def build_graph(train_data):
    graph = nx.DiGraph()  # 定义有向图
    for seq in train_data:  # 遍历session
        for i in range(len(seq) - 1):  # 遍历item
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:  # 如果两个item之间的边不存在，则设置权重为1
                weight = 1
            else:  # 若边存在，则权重加1
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)  # 更新权重
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):  # 查找节点node的入度边（j，i）
            sum += graph.get_edge_data(j, i)['weight']  # 将节点node所有的入度边的权重加起来
        if sum != 0:
            for j, i in graph.in_edges(i):  # 使用edge终点节点的入度用来标准化。论文里说的是除以edge开始节点的出度？？？
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):  # all_usr_pois为输入序列，item_tail为末尾补全数据
    us_lens = [len(upois) for upois in all_usr_pois]  # 生成每一个序列的长度的列表
    len_max = max(us_lens)  # 取出最大长度
    # 将所有序列填充至最长序列的长度len_max，尾部用item_tail补全
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]  # 有序列的位置用1替代，没有动作的位置用0填充。例:us_msks=[[1,1,0],[1,0,0]]
    return us_pois, us_msks, len_max  # 返回用0填充的序列us_pois、mask序列us_msks，和最大序列长度len_max


def split_validation(train_set, valid_portion):  # valid_portion为需要生成的验证集的比例。
    train_set_x, train_set_y = train_set  # train_set是长度为2的元组。包括数据增强后的 序列列表 和对应的 标签列表 两个元素
    n_samples = len(train_set_x)  # 序列个数
    sidx = np.arange(n_samples, dtype='int32')  # 生成一个长度为序列个数的一维数组。[0,1,2,3,...]
    np.random.shuffle(sidx)  # 将sidx的元素随机打乱
    n_train = int(np.round(n_samples * (1. - valid_portion)))  # 计算剩余的训练序列的个数。round为四舍五入函数
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]  # 以n_train为分割线生成验证集和训练集

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)  # 返回训练集格式仍为长度为2的元组类型


class Data():
    def __init__(self, data, shuffle=False, graph=None):  # data：([[1,2],[1],[4]...],[3,2,5...])
        inputs = data[0]  # X，得到输入序列的列表
        inputs, mask, len_max = data_masks(inputs, [0])  # 使用data_masks函数将所有序列用0填充至最大长度
        self.inputs = np.asarray(inputs)  # 补全0后的输入序列，转化为array（）类型（变为矩阵。输入列表中的每个列表为一行，n*len_max）
        self.mask = np.asarray(mask)    # 补全0后的mask序列，转化为array（）类型(n*len_max)
        self.len_max = len_max  # 序列的最大长度（补0后每个序列的长度）
        self.targets = np.asarray(data[1])  # Y，预测的item的列表，变为array类型（shape为n*1）
        self.length = len(inputs)  # 输入序列的个数
        self.shuffle = shuffle   # 是否打乱数据
        self.graph = graph  # 数据图。好像没有用到？

    def generate_batch(self, batch_size):  # 按照batchsize生成每个batch
        if self.shuffle:   # 如果shuffle=True，则对input、mask、targets打乱顺序
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)  # 计算batch的个数
        if self.length % batch_size != 0:
            n_batch += 1   # 向上取整
        slices = np.split(np.arange(n_batch * batch_size), n_batch)  # [arrays([]),arrays([]),...]
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]  # 处理最后一个array,去掉多余的值
        return slices

    def get_slice(self, i):  # 根据索引i得到相应的数据，i为array类型
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]  # 得到对应索引的值
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:  # n_node 存储每个输入序列u_input中item类别数，即去除重复点击项的剩余个数
            n_node.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node)  # 获取item类别最多序列的类别数。其中包括填充的0
        for u_input in inputs:
            node = np.unique(u_input)  # node为当前序列u_input中唯一的item。即去除重复item以后的item
            # 将node变为列表，并将其与最多类别数的序列对齐，用0填充。items是存放这些对齐后序列的列表
            # items：[[max_n_node个值],[max_n_node个值],[max_n_node个值]...]
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))  # max_n_node*max_n_node的零矩阵
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]  # 当前item对应的唯一序列node中的序号
                v = np.where(node == u_input[i + 1])[0][0]  # 下一个item对应的唯一序列node中的序号
                u_A[u][v] = 1  # 矩阵相应位置的值改为1，表示从u_input[i]到u_input[i+1]的次数变为1
            u_sum_in = np.sum(u_A, 0)  # 将矩阵的每列相加，结果为一行
            u_sum_in[np.where(u_sum_in == 0)] = 1  # 将等于0的位置变为1
            u_A_in = np.divide(u_A, u_sum_in)  # u_A除以u_sum_in。广播。得到u_A_in为入度矩阵
            u_sum_out = np.sum(u_A, 1)  # 将矩阵的每行相加，结果为一行
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)  # 同理，得到出度矩阵
            # 将入度矩阵和出度矩阵拼接并转置，u_A相当于论文中的一个session拼接矩阵As
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)  # 将每个输入序列的u_A矩阵添加到列表A中保存
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])  # 点击序列对应唯一item序列集合的位置
        return alias_inputs, A, items, mask, targets
        # 点击序列对应唯一item序列集合的位置；所有session的拼接矩阵的列表；所有session的唯一item对齐后的列表；mask；targets