#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/2/10 下午5:40
  @ Author   : Vodka
  @ File     : SentimentClassification .py
  @ Software : PyCharm
"""
import collections
import os
import random

import torch
import torch.nn as nn
import torchtext.vocab as Vocab

from Dive_into_Deep_Learning.BiRNN_SentimentClassification.BiRNN import BiRNN
from Dive_into_Deep_Learning.BiRNN_SentimentClassification.MyDataset import MyDataset


def tokenize(data):
    """
    将原始数据分词，并转化为小写
    :param data:
    :return:
    """

    def tokenizer(text):
        return [token.lower() for token in text.split(' ')]

    return [tokenizer(context) for context, label in data]


def get_vocab(data):
    """
    获取词汇表，filter：词频需要超过5次
    :param data:
    :return:
    """
    tokenized_data = tokenize(data)
    counter = collections.Counter([token for st in tokenized_data for token in st])
    return Vocab.Vocab(counter, min_freq=5)


def collate(data):
    """
    批量读取数据，使得每条数据以 截断/填充0 的方式变为统一的长度
    :param data:
    :param vocab:
    :return:
    """

    def padding(x):
        if len(x) < max_len:
            return x + [0] * (max_len - len(x))
        else:
            return x[:max_len]

    tokenized_data = tokenize(data)
    contents = torch.tensor([padding([vocab.stoi[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([label for content, label in data])
    return (contents, labels)


def load_pretrained_embedding(words, pretrained_vocab):
    """
    将原vocab_embedding里的words对应的词向量替换为预训练词向量(idx不变，只改变weight.data)
    :param words:
    :param pretrained_vocab:
    :return:
    """
    embedding = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])  # 初始化为0
    cnt_OOV = 0
    for _index, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embedding[_index, :] = pretrained_vocab.vectors[idx]
        except:
            pass
    if cnt_OOV > 0:
        print("No out of vocabulary words.")
    return embedding


def train(data_iter, net, loss, optimizer, device, num_epochs, lr):
    """
    :param data_iter:
    :param net:
    :param loss:
    :param optimizer:
    :param device:
    :param num_epochs:
    :param lr:
    :return:
    """
    net = net.to(device)
    print("Begin training... ")
    for epoch in range(num_epochs):
        l_sum = 0
        n = 0
        for batch in data_iter:
            content, label = [data.to(device) for data in batch]
            pred = net(content)
            l = loss(pred, label)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.item()
            n += 1
        print('epoch %d, loss %.2f' % (epoch + 1, l_sum / n))


def predict(net, device, vocab, sentence):
    """
    对word sentence进行情感分析预测
    :param net:
    :param vocab:
    :param sentence:
    :return:
    """
    sentence = torch.tensor([vocab.stoi[word] for word in sentence]).to(device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)
    if label.item() == 1:
        return 'positive'
    else:
        return 'negative'


# 超参数
batch_size = 64
max_len = 500
embedding_size = 50
num_hiddens = 50
num_layers = 2
learning_rate = 0.01
num_epochs = 1

# 读取数据
train_data = []
for label in ['pos', 'neg']:
    for file in os.listdir('./data_download/train/' + label):
        with open('./data_download/train/' + label + '/' + file, 'rb') as f:
            content = f.read().decode('utf-8').replace('\n', '').replace('<br />', '').lower()
            train_data.append([content, 1 if label == 'pos' else 0])
random.shuffle(train_data)
contents = [content for content, label in train_data]
labels = [label for content, label in train_data]
vocab = get_vocab(train_data)

train_data_set = MyDataset(contents, labels)
train_data_iter = torch.utils.data.DataLoader(train_data_set, batch_size, collate_fn=collate, shuffle=True)

# 设置模型
Net = BiRNN(vocab, embedding_size, num_hiddens, num_layers)
glove_vocab = Vocab.GloVe(name='6B', dim=50, cache='./data_download')
Net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glove_vocab))  # 加载预训练词向量，且设置为在过程中不更新
Net.embedding.weight.requires_grad = False
optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, Net.parameters()), lr=learning_rate)
loss = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练模型
train(data_iter=train_data_iter, net=Net, loss=loss, optimizer=optimizer, device=device, num_epochs=num_epochs,
      lr=learning_rate)
# 预测
print(predict(net=Net, device=device, vocab=vocab, sentence=['i', 'feel', 'bad']))  # negative
print(predict(net=Net, device=device, vocab=vocab, sentence=['the', 'movie', 'is', 'good']))  # positive
print(predict(net=Net, device=device, vocab=vocab, sentence=['how', 'terrible', 'it', 'is']))  # negative
print(predict(net=Net, device=device, vocab=vocab, sentence=['how', 'nice', 'it', 'is']))  # positive
