# Long Short-term Memory，LSTM
长短期记忆（（Long short-term memory）最早是1997年由Hochreiter 和 Schmidhuber在论文《LONG SHORT-TERM MEMORY》中提出的。

LSTM 中引入了3个门，即输入门（input gate）、遗忘门（forget gate）和输出门（outputgate），以及与隐藏状态形状相同的记忆细胞（某些文献把记忆细胞当成一种特殊的隐藏状态），从而记录额外的信息，值域[0,1]。
此外，长短期记忆需要计算候选记忆细胞 。它的计算与上面介绍的3个门类似，但使用了值域在[-1,1]的tanh函数作为激活函数。
我们可以通过元素值域在 的输入门、遗忘门和输出门来控制隐藏状态中信息的流动，这一般也是通过使用按元素乘法来实现的。当前时间步记忆细胞的计算组合了上一时间步记忆细胞和当前时间步候选记忆细胞的信息，并通过遗忘门和输入门来控制信息的流动。有了记忆细胞以后，接下来我们还可以通过输出门来控制从记忆细胞到隐藏状态的信息的流动：从而计算得到了当前时间步的细胞状态C 以及 隐藏状态H。

根据谷歌的测试表明，LSTM中最重要的是Forget gate，其次是Input gate，最次是Output gate。

[参考文章](https://blog.csdn.net/zhangbaoanhadoop/article/details/81952284)