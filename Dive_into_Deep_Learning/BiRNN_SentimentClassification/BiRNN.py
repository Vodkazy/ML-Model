#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/2/10 下午7:40
  @ Author   : Vodka
  @ File     : BiRNN .py
  @ Software : PyCharm
"""
import torch
import torch.nn as nn

class BiRNN(nn.Module):
    def __init__(self,vocab, embedding_size, num_hiddens, num_layers):
        super(BiRNN,self).__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_size)
        self.encoder = nn.LSTM(input_size=embedding_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True ) # 双向RNN
        self.decoder = nn.Linear(4*num_hiddens, 2) # 初始时间步和最终时间步的隐藏状态作为全连接层输⼊

    def forward(self,inputs):
        """
        因为LSTM需要将序列⻓度(seq_len)作为第⼀维，所以将输⼊转置后再提取词特征
        :param inputs: (batch_size * seq_len)
        :return: (seq_len, batch_size, embedding_size)
        """
        embeddings = self.embedding(inputs.permute(1,0))
        outputs,_ = self.encoder(embeddings) # (seq_len, batch_size, 2*num_hidden)
        hidden = torch.cat((outputs[0], outputs[-1]), -1)
        outs = self.decoder(hidden)
        return outs

