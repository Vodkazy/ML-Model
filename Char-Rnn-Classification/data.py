import torch
import torch.nn as nn
import string
import glob
import os
import unicodedata


# return file accordant filenames(including extension filename)
def findFiles(path): return glob.glob(path)


# transfer Unicode to ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in vocabulary
    )


# get names in one txt-file
def readNames(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# get the index of one letter
def getIndex(letter):
    return (vocabulary.find(letter))


# transfer namestringArray to Tensor
def namesToTensor(namestringArray):
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
