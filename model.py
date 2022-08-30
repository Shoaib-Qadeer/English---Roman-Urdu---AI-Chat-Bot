import torch
#parent
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        #hidden size a hidden layer between input and output layer
        #linear function is used to build single layer feed forward network
        self.l1 = nn.Linear(input_size, hidden_size)

        #nn class is used to multiple neural feed forward network
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)

        #activation function to decide whether the neuron should be activated or not /Rectified Linear Unit
        #if input positive direct pass to next layer of nn/ if negative assign 0 output and forward it to
        self.relu = nn.ReLU()



    #Forwarding output of one layer to other

    def forward(self, x):
        out = self.l1(x)

        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)

        return out