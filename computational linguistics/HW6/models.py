
#models.py
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import string

all_letters = string.ascii_letters + " .,;'"
languages = ["af", "cn", "de", "fi", "fr", "in", "ir", "pk", "za"]

'''
    Please add default values for all the parameters of __init__.
'''
class CharRNNClassify(nn.Module):
    def __init__(self, input_size=len(all_letters), hidden_size=128, output_size=len(languages)):

        super(CharRNNClassify, self).__init__()
                 
        self.hidden_size = hidden_size
                 
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()
    
    def forward(self, input, hidden=None):
        lstm_out, hidden = self.lstm(input.view(1,1,-1), hidden)
        out = self.fc1(lstm_out.view(len(input), -1))
        out = F.log_softmax(out, dim=1)
        return out,hidden
    
    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_size),torch.zeros(1, 1, self.hidden_size))

