#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2018/12/17 21:25
  @ Author   : Vodka
  @ File     : datapy.py
  @ Software : PyCharm
"""
import torch
import string
import glob
import os
import unicodedata


def findFiles(path):
    """
    file accordant filenames(including extension filename)
    :param path:
    :return:
    """
    return glob.glob(path)


def unicodeToAscii(s):
    """
    transfer Unicode to ASCII
    :param s:
    :return:
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in vocabulary
    )


def readNames(filename):
    """
    get names in one txt-file
    :param filename:
    :return:
    """
    lines = open(filename).read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


def getIndex(letter):
    """
    get the index of one letter
    :param letter:
    :return:
    """
    return (vocabulary.find(letter))


def namesToTensor(namestringArray):
    """
    transfer namestringArray to Tensor
    :param namestringArray:
    :return:
    """
    # the dimension is:  row:numbers of names  col:57 letters
    names_tensor = torch.zeros(len(namestringArray), 1, len(vocabulary))
    for _index, letter in enumerate(namestringArray):
        names_tensor[_index][0][getIndex(letter)] = 1
    return names_tensor


vocabulary = string.ascii_letters + " .,;'-"
# dict of countries and names
name_dictionary = {}
# list of country names
country_names = []

for filename in findFiles('./data/names/*.txt'):
    country_name = os.path.splitext(os.path.basename(filename))[0]
    country_names.append(country_name)
    names = readNames(filename)
    name_dictionary[country_name] = names

country_size = len(country_names)
vocabulary_size = len(vocabulary)
