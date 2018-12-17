import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN, self).__init__()

        # input to hidden
        self.i2h = nn.Linear(input_size+hidden_size,hidden_size)
        # input to output
        self.i2o = nn.Linear(input_size+hidden_size,output_size)
        # logsoftmax
        self.softmax = nn.LogSoftmax()

    def forward(self,input,hidden):
        combined = torch.cat((input,hidden),1)
        hidden = self.i2h(combined)
        output = self.softmax(self.i2o(combined))
        return output,hidden

    def initHidden(self,hidden_size):
        return (torch.zeros(1,hidden_size))
