#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/2/9 下午10:42
  @ Author   : Vodka
  @ File     : SigmoidBinaryCrossEntropyLoss .py
  @ Software : PyCharm
"""
import torch.nn as nn


class SigmoidBinaryCrossEntropyLoss(nn.Module):
    # 因为进行了负采样，所有需要自定义一个新的损失函数，结合具体的公式进行编写
    def __init__(self):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, mask=None):
        """
        :param inputs: (batch_size, len)
        :param targets:  (batch_size, len)
        :param mask:
        :return:
        """
        inputs, targets, mask = inputs.float(), targets.float(), mask.float()
        loss = nn.functional.binary_cross_entropy_with_logits(inputs,
                                                              targets, reduction="none", weight=mask)
        return loss.mean(dim=1)
