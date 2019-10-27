# Gated Recurrent Unit，GRU
GRU是2014年由Cho, et al在文章《Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation》中提出的，某种程度上GRU也是对于LSTM结构复杂性的优化。LSTM能够解决循环神经网络因长期依赖带来的梯度消失和梯度爆炸问题，但是LSTM有三个不同的门，参数较多，训练起来比较困难。GRU只含有两个门控结构，且在超参数全部调优的情况下，二者性能相当，但是GRU结构更为简单，训练样本较少，易实现。

GRU的结构主要包含重置门和更新门，把GRU看着LSTM的变体，相当于取消了LSTM中的cell state，只使用了hidden state,并且使用update gate更新门来替换LSTM中的输入们和遗忘门，取消了LSTM中的输出门，新增了reset gate重置门。这样做的好处是在达到LSTM相近的效果下，GRU参数更少，训练的计算开销更小，训练速度更快。

由于GRU参数更少，收敛快，通常在数据集够大的情况下，选择LSTM效果应该会更好。
通常情况下LSTM和GRU两者效果相差不大，GRU训练更快，所以一般会先选择使用GRU进行训练和调参，当无法再继续优化时可以把GRU替换成LSTM来看看是否有提高。