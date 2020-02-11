# Attention Model

Google给出注意力机制的定义为，给定一个Query和一系列的key-value对一起映射出一个输出。Attention函数的本质可以被描述为一个查询（query）到一系列（键key-值value）对的映射.
- 将Query与key进行相似性度量(类似于上述的权重wij)
- 将求得的相似性度量进行缩放标准化
- 将权重与value进行加权

在计算attention时主要分为三步，
- 第一步是将query和每个key进行相似度计算得到权重，常用的相似度函数有点积，拼接，感知机等；
- 第二步一般是使用一个softmax函数对这些权重进行归一化；
- 最后将权重和相应的键值value进行加权求和得到最后的attention。目前在NLP研究中，key和value常常都是同一个，即key=value。

Attention主要用在RNN这种具有时序性质的模型中，主要作用就是使得传统的encoder-decoder模型中，在decode时背景向量不是传统的encoder的全部时间步的信息，而是重点突出其中几个的时间步的信息。

[参考文章1](https://www.baidu.com/link?url=iGYWYP1pVLyQBhBc6zr05f-W-ZJEnwfcn-Dd7ZHwbjmohg-bmIAbwh12iZfNS8iu7xpDAh_hrw0bNoqmDIkH4K&wd=&eqid=852021ee0000adb4000000035e4244ab)
[参考文章2](https://www.cnblogs.com/jiangxinyang/p/9367497.html)
[参考文章3](https://zhuanlan.zhihu.com/p/47282410)