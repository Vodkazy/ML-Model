#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/2/11 下午4:18
  @ Author   : Vodka
  @ File     : MachineTranslation .py
  @ Software : PyCharm
"""
import torch.nn as nn

from Dive_into_Deep_Learning.Attention.Simple_attention.Decoder import Decoder
from Dive_into_Deep_Learning.Attention.Simple_attention.Encoder import Encoder
from Dive_into_Deep_Learning.Attention.Simple_attention.Preprocess import *

# 超参数
max_seq_len = 7
embedding_size = 50
num_hiddens = 64
num_layers = 2
attention_size = 10
drop_prob = 0.5
learning_rate = 0.01
batch_size = 16
num_epochs = 100


def batch_loss(encoder, decoder, X, Y, loss):
    """
    :param encoder:
    :param decoder:
    :param X:
    :param Y:
    :param loss:
    :return:
    """
    batch_size = X.shape[0]
    encoder_state = None  # 隐藏态初始化为None时PyTorch会⾃动初始化为0
    encoder_outputs, encoder_state = encoder(X, encoder_state)
    # 直接将编码器最终时间步的隐藏状态作为解码器的初始隐藏状态
    decoder_state = encoder_state
    # 解码器在最初时间步的输入是BOS
    decoder_input = torch.tensor([out_vocab.stoi[BOS]] * batch_size)
    # 使用掩码变量mask来忽略掉标签为填充项PAD的损失, 初始全1
    mask, num_not_pad_tokens = torch.ones(batch_size, ), 0
    l = torch.tensor([0.0])
    for y in Y.permute(1, 0):  # Y shape: (batch, seq_len)
        decoder_output, decoder_state = decoder(decoder_input, decoder_state, encoder_outputs)
        l = l + (mask * loss(decoder_output, y)).sum()
        decoder_input = y  # 使用强制教学
        num_not_pad_tokens += mask.sum().item()
        # EOS后面全是PAD. 下面一行保证一旦遇到EOS接下来的循环中mask就一直是0
        mask = mask * (y != out_vocab.stoi[EOS]).float()
    return l / num_not_pad_tokens


def train(encoder, decoder, dataset, lr, batch_size, num_epochs):
    """
    :param encoder:
    :param decoder:
    :param dataset:
    :param lr:
    :param batch_size:
    :param num_epochs:
    :return:
    """
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    loss = nn.CrossEntropyLoss(reduction='none')
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(num_epochs):
        l_sum = 0.0
        for X, Y in data_iter:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            l = batch_loss(encoder, decoder, X, Y, loss)
            l.backward()
            decoder_optimizer.step()
            encoder_optimizer.step()
            l_sum += l.item()
        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))


def translate(encoder, decoder, input_seq, max_seq_len):
    """
    预测代码直接使用原版代码
    :param encoder:
    :param decoder:
    :param input_seq:
    :param max_seq_len:
    :return:
    """
    in_tokens = input_seq.split(' ')
    in_tokens += [EOS] + [PAD] * (max_seq_len - len(in_tokens) - 1)
    enc_input = torch.tensor([[in_vocab.stoi[tk] for tk in in_tokens]])  # batch=1
    enc_state = None
    enc_output, enc_state = encoder(enc_input, enc_state)
    dec_input = torch.tensor([out_vocab.stoi[BOS]])
    dec_state = enc_state
    output_tokens = []
    for _ in range(max_seq_len):
        dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
        pred = dec_output.argmax(dim=1)
        pred_token = out_vocab.itos[int(pred.item())]
        if pred_token == EOS:  # 当任一时间步搜索出EOS时，输出序列即完成
            break
        else:
            output_tokens.append(pred_token)
            dec_input = pred
    return output_tokens


# 获取输入词的词典、输出词的词典、训练集
in_vocab, out_vocab, dataset = read_data(max_seq_len)
# 设置模型，相当于两个模型，其中Decoder里又套着一个attention模型
encoder = Encoder(len(in_vocab), embedding_size, num_hiddens, num_layers, drop_prob)
decoder = Decoder(len(out_vocab), embedding_size, num_hiddens, num_layers, attention_size, drop_prob)
train(encoder, decoder, dataset, learning_rate, batch_size, num_epochs)
# 预测
input_seq = 'ils regardent !'
label_seq = 'they are watching !'
pred_tokens = translate(encoder, decoder, input_seq, max_seq_len)
label_tokens = label_seq.split(' ')
print('bleu %.3f, predict: %s' % (bleu(pred_tokens, label_tokens, 3), ' '.join(pred_tokens)))
