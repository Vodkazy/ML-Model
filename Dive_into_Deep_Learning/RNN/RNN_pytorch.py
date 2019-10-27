#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2019/10/25 上午11:09
  @ Author   : Vodka
  @ File     : RNN_pytorch .py
  @ Software : PyCharm
"""
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_hiddens = 128
num_steps = 8  # 训练的时候步长是多少 就只能固定的用多长的序列来预测多长的序列 否则就要padding填充
num_epochs = 64  # 过大会过拟合 造成的现象就是 直接背诵原文
batch_size = 16

char_id_dict = {}
id_char_dict = {}


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
        # X的维度是
        yield X, Y


def one_hot(x, n_class, dtype=torch.float32):
    # X shape: (batch), output shape: (batch, n_class)
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
    res.scatter_(1, x.view(-1, 1), 1)
    return res


def to_onehot(X, n_class):
    # X shape: (batch, seq_len),
    # output: seq_len elements of (X.shape[1], batch, n_class)
    return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens, num_layers=1, bidirectional=False)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self.state = None

    def forward(self, inputs, state):
        # X: (seq_len, batch, vocab_size)
        # Y: (seq_len, batch, num_hiddens)
        # output: (num_steps * batch_size, vocab_size)
        X = torch.stack(to_onehot(inputs, vocab_size))  # 这里只能用torch.stack(X)将list转换为tensor
        Y, self.state = self.rnn(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

    def train(self, lr=1e-3):
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        state = None
        for epoch in range(num_epochs):
            torch.cuda.empty_cache()
            print('Epoch: %d' % (epoch + 1))
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
                if (batch_idx + 1) % 10 == 0:
                    print('No.%d batch , Loss: %.3f' % (batch_idx + 1, train_loss / 10))
                    train_loss = 0

        torch.save(self.state_dict(), 'rnn_tangshi.pt')
        print("save model successfully!")

    def predict(self, given_words, seq_len): # 输入规模要和输出规模一致
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
    f = open('tangshi.txt', 'rb')
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

    # 输入形状为(时间步数, 批量大小, 词典大小)
    # 输出形状为(时间步数, 批量大小, 词典大小)
    # 隐藏状态h的形状为(层数, 批量大小, 隐藏单元个数)
    model = RNNModel().to(device)
    # model.train()
    model.load_state_dict(torch.load('rnn_tangshi.pt', map_location=torch.device(device)))
    print(model.predict('金陵月色', 256))

    """
        金陵月色，青山空水雁声。 
        汉河三万里，人大嫁作商人妇。 
        君别意不知，马住，瑶池柳日夕。  
        空文断，秋雨过王看。 
        城中怨野尽，去年，寒禽。 
        不见长安，乐事，铜雀春光销月下。 
        行人有人，不得志沾巾。  
        家生女幽居，偶然回首。 
        长风早晚来。 
        我闻苦死生，坐看红颜老。 
        回看两楹奠，当与梦啼鸟！  
        近边人事，相见，红旗。  
        金阙烟下飞，美人行泪空。 
        圣皇重阳月，相与而言，未有共真未燕。 
        不死生低摇，一片云石万。  
        北阙七月照，清落叶人从初以道之。  
        晚来望君烟，今事非北斗阑干。  
        来文章道已棹，辞歌声花满急，流声一枝弹。
        
        金陵月色风风。 
        明朝散发多秋草，夜飞白发无故乡。  
        不见来者来，双声西畔烟。 
        君言不知其所稀，不见群鸥日日来。 
        江口苍苍。 
        长江春色雪山夜，胡天无事空山来。  
        故园江水谢家君，且乐清光近今出。 
        鸿来不疑倚轻所，空月十五十年十里堤！   
        我欲归无轻敌，始觉声之，烟波澜里谁短心？  
        红从怅望君抱。  
        九回宿舞为华新。 
        盘原上南轻翠接，云雨相送上穷烟，即今无情吊鬓逢？
        夜泊秦明月，自必逢人事暮闻者。 
        五花开元照旧寻，白日登临行雪满。 
        日暮沙草群山东风雨里香。 
        几重更去。故人，以我歌舞水声。  
        露草书书帖似金水寒。
    """
