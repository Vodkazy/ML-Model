# Recurrent Neural Network, RNN
RNN(Recurrent Neural Network)即循环神经网络，用于解决训练样本输入是连续的序列,且序列的长短不一的问题，比如基于时间序列的问题。基础的神经网络只在层与层之间建立了权连接，RNN最大的不同之处就是在层之间的神经元之间也建立的权连接。

RNN之所以称为循环神经网路，即一个序列当前的输出与前面的输出也有关。具体的表现形式为网络会对前面的信息进行记忆并应用于当前输出的计算中，即隐藏层之间的节点不再无连接而是有连接的，并且隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。

当前时间步t的隐藏状态H_t将参与下一个时间步t+1的隐藏状态H_t+1的计算，并输入到当前时间步的全连接输出层。

RNN主要用来解决序列问题，强调的是先后顺序，在NLP中引申出上下文的概念，一个翻译问题，这个词的含义可能和前后的单词形成的这个组合有联系（Skip-gram）,也可能是它之前的所有单词都有联系（Attention），并且，借助RNN的state这样的记忆单元，使得一个序列位置的输出在数学上和之前的所有序列的输入都是有关系的。当然原始的RNN由于梯度的乘性问题，前面的序列的影响近乎为0，这个后面又用LSTM来修正为加性问题。RNN的数学基础可以认为是马尔科夫链，认为后续的值是有前者和一些参数的概率决定的。

RNN优点：可以拟合序列数据，（LSTM）通过遗忘门和输出门忘记部分信息来解决梯度消失的问题。缺点：梯度消失；无法很好的并行（工业上影响很大）

[参考文章](https://www.jianshu.com/p/53e457937557)