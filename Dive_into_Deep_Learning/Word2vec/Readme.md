# Word2vec embedding

word2vec 是 Google 于 2013 年开源推出的一个用于获取 word vector 的工具包，它简单、高效，因此引起了很多人的关注。
word2vec主要包括两个模型，一个是CBOW（利用背景词生成中心词），一个是Skip-gram（利用中心词生成背景词），训练时需要利用负采样来减少语料库中词频分布差异过大带来的影响
word2vec的参数有两个参数矩阵，一个是中心词embedding矩阵，一个是背影词embedding矩阵，Skip-gram用的是中心词矩阵，CBOW用的是背景词矩阵
此部分用到了**Mask掩码**

[参考文章1](https://www.cnblogs.com/peghoty/p/3857839.html)
[参考文章2](https://www.zybuluo.com/Dounm/note/591752)