import torch
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

# basic classifier model
class MLP(nn.Module):

    def __init__(self, input_size, hiddens, output_size, device=None):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.dim_list = [input_size, *hiddens, output_size]
        self.device = device

        if device != None:
            torch.cuda.set_device(device)

        self.layers = []
        for (dim1, dim2) in zip(self.dim_list[:-2], self.dim_list[1:-1]):
            if self.device != None:
                self.layers.append(nn.Linear(dim1, dim2).cuda())
                self.layers.append(nn.Tanh().cuda())
            else:
                self.layers.append(nn.Linear(dim1, dim2))
                self.layers.append(nn.Tanh())
        if self.device != None:
            self.layers.append(nn.Linear(self.dim_list[-2], self.dim_list[-1]).cuda())
        else:
            self.layers.append(nn.Linear(self.dim_list[-2], self.dim_list[-1]))
        
        self.models = nn.Sequential(*self.layers)

    def forward(self, input):
    
        assert(input.shape[1] == self.input_size)

        if self.device != None:
            torch.cuda.set_device(self.device)
            input = input.cuda()
        
        output = self.models(input)
        return output
    
    def to_cpu(self):
        self.device = None
        for model in self.models:
            model = model.cpu()
    
    def to_cuda(self, device):
        self.device = device
        torch.cuda.set_device(self.device)
        for model in self.models:
            model = model.cuda()
