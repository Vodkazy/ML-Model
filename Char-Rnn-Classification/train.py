import torch
import time
import random
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from model import *
from data import *

hidden_size = 128
epoch_size = 200000
print_point = 5000  # processing status will be printed into screen at print_point
plot_point = 1000   # mark one point in matplot picture
learning_rate = 0.005

# return the most likelihood country when given the output of prediction
def getMostlikelyCountry(output):
    # Get top N categories
    max_value, max_pos = output.data.topk(1)
    name_index = max_pos[0][0]
    return country_names[name_index], name_index

# choose one name from given dataset randomly
def randomChooseOneName(names):
    return names[random.randint(0,len(names)-1)]

# choose one country and its names from given dataset randomly
def randomChooseTrainingData():
    country_name = randomChooseOneName(country_names)
    names = randomChooseOneName(name_dictionary[country_name])
    country_name_tensor = torch.LongTensor([country_names.index(country_name)])
    names_tensor = namesToTensor(names)
    return country_name, names, country_name_tensor, names_tensor

# calculate time from begin_time to now
def timing(begin_time):
    now = time.time()
    s = now - begin_time
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# train the RNN model
def train(country_name_tensor, names_tensor):
    hidden = rnn.initHidden(hidden_size)
    optimizer.zero_grad()

    # everytime we use a name to train
    for i in range(names_tensor.size()[0]):
        output, hidden = rnn(names_tensor[i], hidden)

    # loss is also a tensor
    loss = criterion(output, country_name_tensor)
    loss.backward()

    optimizer.step()
    return output, loss.data[0]

# save out model as one file
def saveModel(model,filename):
    torch.save(model, filename)

# print the matplot picture
def plotDraw(datas):
    plt.figure()
    plt.plot(datas)
    plt.show()

current_loss = 0
all_losses = []

rnn = RNN(vocabulary_size, hidden_size, country_size)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

start = time.time()

# Iterative training
for epoch in range(1, epoch_size + 1):
    country_name, names, country_name_tensor, names_tensor = randomChooseTrainingData()
    output, loss = train(country_name_tensor, names_tensor)
    current_loss += loss

    if epoch % print_point == 0:
        guess_country, guess_country_index = getMostlikelyCountry(output)
        ans = '✓' if guess_country == country_name else '✗ (%s)' % country_name
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / epoch_size * 100, timing(start), loss, names, guess_country, ans))

    if epoch % plot_point == 0:
        all_losses.append(current_loss / plot_point)
        current_loss = 0

saveModel(rnn, 'char-rnn-classification.pt')
plotDraw(all_losses)
