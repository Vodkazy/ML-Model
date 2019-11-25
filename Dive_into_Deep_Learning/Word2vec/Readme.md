# Word2vec embedding

word2vec 是 Google 于 2013 年开源推出的一个用于获取 word vector 的工具包，它简单、高效，因此引起了很多人的关注。由于 word2vec 的作者 Tomas Mikolov 在两篇相关的论文 [3,4] 中并没有谈及太多算法细节，因而在一定程度上增加了这个工具包的神秘感。一些按捺不住的人于是选择了通过解剖源代码的方式来一窥究竟。
word2vec主要包括两个模型，一个是CBOW，一个是Skip-gram.

[参考文章1](https://www.cnblogs.com/peghoty/p/3857839.html)
[参考文章2](https://www.zybuluo.com/Dounm/note/591752)