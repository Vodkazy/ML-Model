# Softmax Regression softmax回归
Softmax回归（Softmax Regression）又被称作多项逻辑回归（multinomial logistic regression），它是logistic回归模型在多分类问题上的推广，即类标签可取两个以上的值。适用于分类问题。它使用softmax运算输出类别的概率分布。
softmax回归是一个单层神经网络，输出个数等于分类问题中的类别个数。
在分类问题中，softmax函数常常和交叉熵损失函数一起使用，其中交叉熵适合衡量两个概率分布的差异。