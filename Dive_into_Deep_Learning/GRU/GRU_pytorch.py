#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/10/27 下午3:52
  @ Author   : Vodka
  @ File     : GRU_pytorch .py
  @ Software : PyCharm
"""
import math
import time
import torch
from torch import nn

num_hiddens = 128
num_steps = 8  # 训练的时候步长是多少 就只能固定的用多长的序列来预测多长的序列 否则就要padding填充
num_epochs = 40  # 过大会过拟合 造成的现象就是 直接背诵原文
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
char_id_dict = {}
id_char_dict = {}

"""
    我们通常使用困惑度（perplexity）来评价语言模型的好坏。显然，任何一个有效模型的困惑度必须小于类别个数.
    困惑度是对交叉熵损失函数做指数运算后得到的值。特别地，
        - 最佳情况下，模型总是把标签类别的概率预测为1，此时困惑度为1；
        - 最坏情况下，模型总是把标签类别的概率预测为0，此时困惑度为正无穷；
        - 基线情况下，模型总是预测所有类别的概率都相同，此时困惑度为类别个数
"""


# 数据处理
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=device):
    # 每次处理batch_size个字符，需处理batch_len次，batch_size可看做有batch_size个程序并行处理
    # 每num_steps个字符构成一个句子，当做一组；那么一共有epoch_size组
    # 也就是说，一共有epoch_size组训练数据，每组里面有num_steps个单词，每次同时处理batch_size个训练数据，一共就有batch_size×epoch_size×num_steps个字符
    # 那么返回的每个X都是 batch_size × num_steps 维的
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y


# 转换实数为one-hot向量
def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


# 转换多个实数为多个one-hot向量
def to_onehot(X, n_class):
    # X shape: (batch, seq_len),
    # output: seq_len elements of (X.shape[1], batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens, num_layers=1, bidirectional=False)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self.state = None

    # 前向传播
    def forward(self, inputs, state):
        # X: (seq_len, batch, vocab_size)
        # Y: (seq_len, batch, num_hiddens)
        # output: (num_steps * batch_size, vocab_size)
        X = torch.stack(to_onehot(inputs, vocab_size))  # 这里只能用torch.stack(X)将list转换为tensor
        Y, self.state = self.gru(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

    # 训练
    def train(self, lr=1e-3):
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        state = None
        for epoch in range(num_epochs):
            l_sum, n, start = 0.0, 0, time.time()
            torch.cuda.empty_cache()
            train_loss = 0
            dataset = data_iter_consecutive(corpus_indices, batch_size, num_steps, device)
            for batch_idx, (X, Y) in enumerate(dataset):
                if state is not None:
                    # 使用detach函数从计算图分离隐藏状态, 这是为了
                    # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                    if isinstance(state, tuple):  # LSTM, state:(h, c)
                        state = (state[0].detach(), state[1].detach())
                    else:
                        state = state.detach()
                optimizer.zero_grad()
                X, Y = X.to(device), Y.to(device)
                output, state = self(X, state)
                # Y的形状是(batch_size, num_steps)，转置后再变成长度为 batch * num_steps 的向量，这样跟输出的行一一对应
                y = torch.transpose(Y, 0, 1).contiguous().view(-1)
                l = loss(output, y.long())
                l.backward()
                optimizer.step()
                train_loss += l.item()
                l_sum += l.item() * y.shape[0]
                n += y.shape[0]
                if (batch_idx + 1) % 10 == 0:
                    print('No.%d batch , Loss: %.3f' % (batch_idx + 1, train_loss / 10))
                    train_loss = 0
            try:
                perplexity = math.exp(l_sum / n)
            except OverflowError:
                perplexity = float('inf')
            print('Epoch %d, perplexity %f, time %.2f sec' % (epoch + 1, perplexity, time.time() - start))
        # 保存模型
        torch.save(self.state_dict(), 'rnn_tangshi.pt')
        print("save model successfully!")

    # 预测
    def predict(self, given_words, seq_len):  # 输入规模要和输出规模一致
        state = None
        output = [char_id_dict[given_words[0]]]
        for t in range(seq_len + len(given_words) - 1):
            X = torch.tensor([output[-1]], device=device).view(1, 1)
            if state is not None:
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].to(device), state[1].to(device))
                else:
                    state = state.to(device)
            (Y, state) = self(X, state)  # 前向计算不需要传入模型参数
            if t < len(given_words) - 1:
                output.append(char_id_dict[given_words[t + 1]])
            else:
                output.append(int(Y.argmax(dim=1).item()))
        return ''.join([id_char_dict[i] for i in output])


if __name__ == '__main__':
    # 读取数据
    f = open('../RNN/tangshi.txt', 'rb')
    corpus = f.read().decode('utf-8').replace("\n", " ").replace("\r", " ")
    _index = 0
    for _, value in enumerate(corpus):
        if value not in char_id_dict.keys():
            char_id_dict[value] = _index
            _index += 1
    for _key, _value in char_id_dict.items():
        id_char_dict[_value] = _key
    vocab_size = len(char_id_dict)
    corpus_indices = [char_id_dict[char] for char in corpus]

    model = GRUModel().to(device)
    model.train()
    model.load_state_dict(torch.load('rnn_tangshi.pt', map_location=torch.device(device)))
    print(model.predict('金陵月色', 256))

    """
        金陵月色，不见功名！  
        君不见，将军下马客，百事！ 
        一枝春风无限，因休露草黄鹤楼。  
        君不见，将军在纸，万事苍茫接大江流。 
        云鬓花颜金步摇，芙蓉帐暖度春宵。 
        金阙西厢叩玉扃，转教小玉报双成。 
        闻道欲来相对此，汉使人语空知心。  
        桂魄初生秋阳斜，汉使人语未识开。 
        其鸣昼已矣，鸟道不是赏，长飙风吹不度愁。 
        今夜偏知人，今日垂杨生。  
        君不见，金粟堆前松柏里。龙媒去尽未成！  
        三十年征戍之，不敢问来发。  
        楚客三峡里，云山况是客。  
        晚来白发，空吟感我至今身。  
        今日暮乡皆人未，李阴阴阴夏水田。
    """
