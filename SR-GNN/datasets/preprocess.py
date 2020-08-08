#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import argparse  # 在命令行控制参数输入的库
import time
import csv
import pickle
import operator
import datetime
import os

# 使用argparse创建一个ArgumentParser对象，其包含了将命令行解析出python数据类型所需的全部信息
parser = argparse.ArgumentParser()
# 给ArgumentParser添加程序参数信息。这些信息在parse_arge()调用时被存储和使用。
# 选用数据集，'--dataset'表示可选项，默认值是'sample'
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose/sample')
opt = parser.parse_args()
print(opt)

dataset = 'sample_train-item-views.csv'
if opt.dataset == 'diginetica':
    dataset = 'train-item-views.csv'  # diginetica数据集的文件名
elif opt.dataset =='yoochoose':
    dataset = 'yoochoose-clicks.dat'  # yoochoose数据集的文件名

print("-- Starting @ %ss" % datetime.datetime.now())
with open(dataset, "r") as f:
    if opt.dataset == 'yoochoose':   # yoochoose数据集每行为：Session ID，Timestamp，Item ID，Category
         # yoochoose数据集是用','分隔的，另外两个都是用';'.原始yoochoose数据集没有表头，需要手动添加上去
        reader = csv.DictReader(f, fieldnames=['session_id', 'timestamp', 'item_id', 'category'], delimiter=',')
    else:
        reader = csv.DictReader(f, delimiter=';')  # DictReader返回的数据类型是csv.DictReader类型的数据
    sess_clicks = {}   # session id 为key，item作为value
    sess_date = {}  # 使用session id作为key，日期时间作为value的字典
    ctr = 0
    curid = -1  # 存放当前id
    curdate = None  # 存放当前日期
    # for循环生成sess_clicks和sess_date两个字典
    for data in reader:  # reader中的行类型是orderedDict类型,如：{'session_id':2,'user_id':....}
        sessid = data['session_id']
        if curdate and not curid == sessid:  # 如果当前session id和读取的session id不相等且curdata存在，则执行if下的内容
            date = ''
            if opt.dataset == 'yoochoose':
                date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))  # date为用秒表示时间的浮点数，例：1462723200.0
            else:
                date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))  # 将时间字符串转换成时间元组，然后变为用秒表示的浮点数
            sess_date[curid] = date   # 例：｛1：1462723200.0，2：1460217600.0｝这里的date只会记录每个session第一个item的时间
        curid = sessid
        if opt.dataset == 'yoochoose':
            item = data['item_id']
        else:
            item = data['item_id'], int(data['timeframe'])  # diginetica的item是一个包括item_id和timeframe的列表
            # item: ('81766',526309),tuple类型
        curdate = ''
        if opt.dataset == 'yoochoose':
            curdate = data['timestamp']
        else:
            curdate = data['eventdate']

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]  # 例：{'1': [('81766', 526309), ('31331', 1031018)]}
        else:
            sess_clicks[sessid] = [item]
        ctr += 1
    date = ''
    if opt.dataset == 'yoochoose':
        # yoochoose数据集每个session的item本来就是按时间排好序的，不需要再次排序
        date = time.mktime(time.strptime(curdate[:19], '%Y-%m-%dT%H:%M:%S'))
    else:
        date = time.mktime(time.strptime(curdate, '%Y-%m-%d'))
        for i in list(sess_clicks):  # list(sess_clicks)的值是由key组成的列表
            sorted_clicks = sorted(sess_clicks[i], key=operator.itemgetter(1))  # 对sess_click的每一个key内，根据timeframe从小到大排序
            sess_clicks[i] = [c[0] for c in sorted_clicks]  # 将排好序的item_id放到对应的sess_click[i]中。sess_click的value只包括item_id
    sess_date[curid] = date  # 对最后一个session添加信息到sess_date
print("-- Reading data @ %ss" % datetime.datetime.now())
# 经过上述处理，生成了两个字典sess_clicks和sess_date，其中sess_clicks的value值是item_id，且已经根据点击时间排好序
#例：sess_clicks:{'1':['9654','33043','32118','12352','35077','36118','81766','129055','31331','32627'],'2':[...]}
#    sess_data:{'1':14627232000.0,'2':14627232000.0,'3',14627232000.0,...}

# Filter out length 1 sessions 过滤掉长度为1的session
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}  # 用来存放每个item出现的次数。key为item_id,value为其出现的次数
for s in sess_clicks:  # 这里的s为字典的key值
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1
# 经过上述循环，得到iid_counts,例：iid_counts={'9654': 4, '33043': 2, '32118': 1, '12352': 2, '35077': 2}

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))  # 对iid_counts根据item出现次数从小到大排序

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= 5, curseq))  # 从corseq中，筛选出总出现次数不少于5次的item
    if len(filseq) < 2:   # 如果一个session中总出现次数不少于5次的item的个数小于2，则筛掉该session
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq
# 经过以上筛选，选出了出现次数不少于5次的item，并且过滤掉了这种item数量小于2的session
# 总的来说，经过两步筛选，得到的最终结果为session中的每个item总出现次数都不少于5次，且session长度大于1


# Split out test set based on dates
dates = list(sess_date.items())  # 每个键值对是列表的一个元素
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date  # 找到时间最大（最新）的时间

# 7 days for test
splitdate = 0
if opt.dataset == 'yoochoose':
    splitdate = maxdate - 86400 * 1  # the number of seconds for a day：86400
else:
    splitdate = maxdate - 86400 * 7  # 划分日期从倒数第七天开始，即最后七天

print('Splitting date', splitdate)      # Yoochoose: ('Split date', 1411930799.0)
tra_sess = filter(lambda x: x[1] < splitdate, dates)  # 生成的是一个可迭代对象
tes_sess = filter(lambda x: x[1] > splitdate, dates)  # 根据划分日期将数据集划分为训练集合测试集

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # 按照日期排序，得到的结果格式和date相同，例：[(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print('Length of train set:',len(tra_sess))    # 186670    # 7966257
print('Length of test set:',len(tes_sess))    # 15979     # 15324
print(tra_sess[:3])
print(tes_sess[:3])
print("-- Splitting train set and test set @ %ss" % datetime.datetime.now())

# Choosing item count >=5 gives approximately the same number of items as reported in paper  计算筛选后的item个数
item_dict = {}  # item字典，key为item原编号，value为从1开始的数字编号。
# Convert training sessions to sequences and renumber items to start from 1
def obtian_tra():
    train_ids = []
    train_seqs = []
    train_dates = []
    item_ctr = 1
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:  # 如果当前item i 已经在字典中存在，则outseq从字典中找到i的编号
                outseq += [item_dict[i]]
            else:
                outseq += [item_ctr]  # 等价于 outset.append(item_ctr)
                item_dict[i] = item_ctr
                item_ctr += 1  # 如果当前 item 在item_dict中没有记录，则表示这是一个新的item，编号+1
        if len(outseq) < 2:  # Doesn't occur  之前已经将长度小于2的session筛选出去了，所以这个判断不会为真
            continue
        train_ids += [s]
        train_dates += [date]
        train_seqs += [outseq]
    print('Number of train items:',item_ctr)     # 43098, 37484
    return train_ids, train_dates, train_seqs
    # train_ids 是保存所有session id 的列表。train_dates保存对应的时间，train_seqs保存每个session中的item编号，这里的item编号从1重新编号

# Convert test sessions to sequences, ignoring items that do not appear in training set
# 将测试集也转换成序列，并且忽略掉在训练集中没出现过的item
def obtian_tes():
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for i in seq:
            if i in item_dict:  # 若当前item i没有在训练集的item_dict中，则忽略掉此item
                outseq += [item_dict[i]]
        if len(outseq) < 2:
            continue
        test_ids += [s]
        test_dates += [date]
        test_seqs += [outseq]
    return test_ids, test_dates, test_seqs


tra_ids, tra_dates, tra_seqs = obtian_tra()
tes_ids, tes_dates, tes_seqs = obtian_tes()

def process_seqs(iseqs, idates):  # 使用数据增强的方法，将每个session分成多个seq-label对，比如[(v1),v2]、[(v1,v2),v3]等
    out_seqs = []
    out_dates = []
    labs = []
    ids = []  # 这里将session编号变为从0开始的整数
    # zip函数将可迭代对象作为参数，将对象中对应的元素打包成元组然后返回这些元组组成的列表。返回值是可迭代对象
    for id, seq, date in zip(range(len(iseqs)), iseqs, idates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]  # 等价于 out_seqs.append([seq[:-i]])
            out_dates += [date]
            ids += [id]
    return out_seqs, out_dates, labs, ids
    # out_seqs表示序列，out_dates表示对应的每个序列的日期，labs对应每个序列的'label',ids对应序列位于的session的编号（从0开始）
    # 比如，对于session：[v1,v2,v3],生成的out_seqs：[v1]、[v1,v2].对应的labs为：v2、v3.
    # out_seqs, out_dates, labs, ids的长度一致
tr_seqs, tr_dates, tr_labs, tr_ids = process_seqs(tra_seqs, tra_dates)
te_seqs, te_dates, te_labs, te_ids = process_seqs(tes_seqs, tes_dates)
tra = (tr_seqs, tr_labs)  # tra是长度为2的元组
tes = (te_seqs, te_labs)
print('数据增强后的训练序列个数',len(tr_seqs))  # 数据增强后的训练序列个数
print('数据增强后的测试序列个数',len(te_seqs))
print(tr_seqs[:3], tr_dates[:3], tr_labs[:3])
print(te_seqs[:3], te_dates[:3], te_labs[:3])
all = 0  # all表示训练集和测试集所有session的长度和

for seq in tra_seqs:
    all += len(seq)
for seq in tes_seqs:
    all += len(seq)
print('avg length: ', all/(len(tra_seqs) + len(tes_seqs) * 1.0))
if opt.dataset == 'diginetica':
    if not os.path.exists('diginetica'):
        os.makedirs('diginetica')
    pickle.dump(tra, open('diginetica/train.txt', 'wb'))  # dump将打包好的tra，写入文件中（第二个参数），这个文件必须有一个write方法
    pickle.dump(tes, open('diginetica/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('diginetica/all_train_seq.txt', 'wb'))  # tra_seqs元素是每个session的item列表。长度是总session个数。
elif opt.dataset == 'yoochoose':
    if not os.path.exists('yoochoose1_4'):
        os.makedirs('yoochoose1_4')
    if not os.path.exists('yoochoose1_64'):
        os.makedirs('yoochoose1_64')
    pickle.dump(tes, open('yoochoose1_4/test.txt', 'wb'))
    pickle.dump(tes, open('yoochoose1_64/test.txt', 'wb'))  # yoochoose两个子集使用同一个测试集

    split4, split64 = int(len(tr_seqs) / 4), int(len(tr_seqs) / 64)  # 数据增强后的长度再取1/4、1/64
    print(len(tr_seqs[-split4:]))
    print(len(tr_seqs[-split64:]))

    tra4, tra64 = (tr_seqs[-split4:], tr_labs[-split4:]), (tr_seqs[-split64:], tr_labs[-split64:])
    seq4, seq64 = tra_seqs[tr_ids[-split4]:], tra_seqs[tr_ids[-split64]:]

    pickle.dump(tra4, open('yoochoose1_4/train.txt', 'wb'))
    pickle.dump(seq4, open('yoochoose1_4/all_train_seq.txt', 'wb'))

    pickle.dump(tra64, open('yoochoose1_64/train.txt', 'wb'))
    pickle.dump(seq64, open('yoochoose1_64/all_train_seq.txt', 'wb'))

else:
    if not os.path.exists('sample'):
        os.makedirs('sample')
    pickle.dump(tra, open('sample/train.txt', 'wb'))
    pickle.dump(tes, open('sample/test.txt', 'wb'))
    pickle.dump(tra_seqs, open('sample/all_train_seq.txt', 'wb'))

print('Done.')
