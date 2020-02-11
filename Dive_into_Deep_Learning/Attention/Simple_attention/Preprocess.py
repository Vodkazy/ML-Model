#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/2/11 下午4:28
  @ Author   : Vodka
  @ File     : Preprocess .py
  @ Software : PyCharm
"""
import collections
import io
import math

import torch
import torch.utils.data as Data
import torchtext.vocab as Vocab

"""
    以下代码直接使用原书提供的代码
"""

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'


def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    """
    将一个序列中所有的词记录在all_tokens中以便之后构造词典，然后在该序列后面添加PAD直到序列,长度变为max_seq_len，然后将序列保存在all_seqs中
    :param seq_tokens:
    :param all_tokens:
    :param all_seqs:
    :param max_seq_len:
    :return:
    """
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)


def build_data(all_tokens, all_seqs):
    """
    使用所有的词来构造词典。并将所有序列中的词变换为词索引后构造Tensor
    :param all_tokens:
    :param all_seqs:
    :return:
    """
    vocab = Vocab.Vocab(collections.Counter(all_tokens),
                        specials=[PAD, BOS, EOS])
    indices = [[vocab.stoi[w] for w in seq] for seq in all_seqs]
    return vocab, torch.tensor(indices)


def read_data(max_seq_len):
    """
    读取数据
    :param max_seq_len:
    :return:
    """
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    with io.open('./data/fr-en-small.txt') as f:
        lines = f.readlines()
    for line in lines:
        in_seq, out_seq = line.rstrip().split('\t')
        in_seq_tokens, out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
        if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
            continue  # 如果加上EOS后长于max_seq_len，则忽略掉此样本
        process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
        process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
    in_vocab, in_data = build_data(in_tokens, in_seqs)
    out_vocab, out_data = build_data(out_tokens, out_seqs)
    return in_vocab, out_vocab, Data.TensorDataset(in_data, out_data)


def bleu(pred_tokens, label_tokens, k):
    """
    机器翻译评价指标
    :param pred_tokens:
    :param label_tokens:
    :param k:
    :return:
    """
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[''.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[''.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
