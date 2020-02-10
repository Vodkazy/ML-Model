#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/2/8 下午9:21
  @ Author   : Vodka
  @ File     : Skip-gram .py
  @ Software : PyCharm
"""
import collections
import math
import random

import torch
import torch.nn as nn

from Dive_into_Deep_Learning.Word2vec.MyDataset import MyDataset
from Dive_into_Deep_Learning.Word2vec.SigmoidBinaryCrossEntropyLoss import SigmoidBinaryCrossEntropyLoss


def read_data(path):
    """
    读取数据
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        lines = f.readlines()
        raw_dataset = [st.split() for st in lines]
    counter = collections.Counter([token for st in raw_dataset for token in st])
    counter = dict(filter(lambda x: x[1] >= 5, counter.items()))
    idx_to_token = [token for token, num in counter.items()]  # [idx] = token
    token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}  # [token] = idx
    dataset = [[token_to_idx[token] for token in st if token in token_to_idx] for st in raw_dataset]
    return dataset, idx_to_token, token_to_idx, counter


def subsample_dataset(dataset):
    """
    原始数据集词频分布差距太大 容易对结果产生影响 因此需要二次采样
    :param dataset:
    :return:
    """
    total_token_num = sum([len(st) for st in dataset])  # 训练集中所有的token数量
    new_dataset = [[idx for idx in st
                    if random.uniform(0, 1) > 1 - math.sqrt(1e-4 / counter[idx_to_token[idx]] * total_token_num)]
                   # 以右侧概率丢弃该词，越常见词越容易被丢弃从而减少词频
                   for st in dataset]
    return new_dataset


def get_centers_and_contexts(dataset, max_window_size):
    """
    提取中间词 以及其对应的背景词
    :param dataset:
    :param max_window_size:
    :return:
    """
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:  # 每个句子至少要有2个词才可能组成一对“中心词-背景词”
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)  # 将中心词排除在背景词之外
            contexts.append([st[idx] for idx in indices])
    return centers, contexts


def get_negatives(contexts, K):
    """
    负采样
    :param contexts:
    :param K:
    :return:
    """
    sampling_weights = [counter[w] ** 0.75 for w in idx_to_token]  # 每个词的权重，词频的0.75次方

    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:  # 每个背景词都要随机采样K个噪声词
            if i == len(neg_candidates):
                # 为了高效计算，可以将k设得稍大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪声词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


def collate(data):
    """
    用作DataLoader的参数collate_fn: 输入是个长为batchsize的list, list中的每个元素都是__getitem__得到的结果
    :param data:
    :return: 每回返回一个四元列表
    """
    max_len = max(len(context) + len(negative) for center, context, negative in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]  # 用0补全成定长
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]  # 补全的0掩码设置为0，其余为1.目的是为了避免填充项对损失函数计算的影响
        labels += [[1] * len(context) + [0] * (max_len - len(context))]  # 正例为1，负例为0
    return (torch.tensor(centers).view(-1, 1), torch.tensor(contexts_negatives),
            torch.tensor(masks), torch.tensor(labels))


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    """
    跳字模型
    :param center: batch_size × 1
    :param contexts_and_negatives: batch_size × max_len
    :param embed_v: 中心词embedding矩阵
    :param embed_u: 背景词embedding矩阵
    :return: batch_size × 1 × max_len ， 输出中的每个元素是中心词向量与背景词向量或噪声词向量的内积
    """
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


def train(net, loss, optimizer, device, num_epochs, lr):
    net = net.to(device)
    print("Begin training... ")
    for epoch in range(num_epochs):
        l_sum = 0
        n = 0
        for batch in data_iter:
            center, context_negative, mask, label = [data.to(device) for data in batch]
            pred = skip_gram(center, context_negative, net[0], net[1])
            # 使用掩码变量mask来避免填充项对损失函数计算的影响，mask在计算loss时起作用
            l = (loss(pred.view(label.shape), label, mask) *
                 mask.shape[1] / mask.float().sum(dim=1)).mean()  # 用⼀个batch的平均loss来当做本批量的误差
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item()
            n += 1
        print('epoch %d, loss %.2f' % (epoch + 1, l_sum / n))


# 超参数
batch_size = 256
embedding_size = 50
num_epochs = 20
learning_rate = 0.01

# 初始数据
dataset, idx_to_token, token_to_idx, counter = read_data('./data/ptb.train.txt')
dataset = subsample_dataset(dataset)
centers, contexts = get_centers_and_contexts(dataset, 5)
negatives = get_negatives(contexts, 5)

# 转化为迭代式读取方式
dataset = MyDataset(centers, contexts, negatives)
data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate, num_workers=4)

# 定义模型
Net = nn.Sequential(
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embedding_size),
    nn.Embedding(num_embeddings=len(idx_to_token), embedding_dim=embedding_size)  # 嵌入层的输入为数字(索引)tensor，输出为其对应的向量
)
loss = SigmoidBinaryCrossEntropyLoss()
optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train(net=Net, loss=loss, optimizer=optimizer, device=device, num_epochs=num_epochs, lr=learning_rate)
word2vec = Net[0].weight.data
print(word2vec[token_to_idx['have']])

"""
总结：
    1.可以使用PyTorch通过负采样训练跳字模型。
    2.二次采样试图尽可能减轻高频词对训练词嵌入模型的影响。
    3.可以将长度不同的样本填充至长度相同的小批量，并通过掩码变量区分非填充和填充，然后只令非填充参与损失函数的计算。
"""