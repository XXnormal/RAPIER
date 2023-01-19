import torch
import torch.nn as nn 
import math
from torch.nn import functional as F
import numpy as np 

class GEN(nn.Module):

    def __init__(self, input_size, hiddens, output_size, device=None):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.dim_list = [input_size, *hiddens, output_size]
        self.device = device

        if device != None:
            torch.cuda.set_device(device)

        self.layers = []
        for (dim1, dim2) in (self.dim_list[:-2], self.dim_list[1:-1]):
            if self.device != None:
                self.layers.append(nn.Linear(dim1, dim2).cuda())
                self.layers.append(nn.ReLU().cuda())
            else:
                self.layers.append(nn.Linear(dim1, dim2))
                self.layers.append(nn.ReLU())
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
            
class MLP(nn.Module):
    
    def __init__(self, input_size, hiddens, output_size, device=None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.dim_list = [input_size, *hiddens, output_size]
        self.device = device

        self.alpha_value = None
        self.mu_value = None
        self.sigma_value = None
        self.lambda_value = None

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

        assert (input.shape[1] == self.input_size)

        if self.device != None:
            torch.cuda.set_device(self.device)
            input = input.cuda()

        output = self.models(input)
        return output

    def f(self, input):
        assert (input.shape[1] == self.input_size)

        if self.device != None:
            torch.cuda.set_device(self.device)
            input = input.cuda()

        output = input
        for i in range(len(self.models)):
            output = self.models[i](output)
            if i == 3:
                break

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



    # Gaussian distribution probability calculation
    def calculate_gau_pro(self, score, mu_value, sigma_value):
        return torch.exp(-((score - mu_value) ** 2) / (2 * (sigma_value ** 2))) / (torch.sqrt(2 * torch.tensor(math.pi)) * sigma_value)

    # Exponential distribution probability calculation
    def calculate_exp_pro(self, score, lambda_value):
        return lambda_value * torch.exp(-lambda_value * score)

    # Probability of equation 15
    def calculate_pro_equ15(self, score, alpha_value, mu_value, sigma_value, lambda_value):
        gau_pro = self.calculate_gau_pro(score, mu_value, sigma_value)
        exp_pro = self.calculate_exp_pro(score, lambda_value)
        return alpha_value * gau_pro / (alpha_value * gau_pro + (1 - alpha_value) * exp_pro)

    # Get the alpha mu sigma and lambda parameters with given labels to minimize the LL function
    # Based on Equation 17 18 19 and 20
    def get_minimize_parameters(self, current_target, scores):
        t = current_target
        f = scores

        mu_value = torch.sum(torch.multiply(t, f)) / torch.sum(t)
        sigma_value = torch.sum(torch.multiply(t, torch.pow(f - mu_value, 2))) / torch.sum(t)
        lambda_value = torch.sum(t) / torch.sum(torch.multiply(t, f))
        alpha_value = torch.mean(t)

        return alpha_value, mu_value, sigma_value, lambda_value

    # Initialize the parameters of alpha mu sigma and lambda
    # For simplity, we use 0.5, 0, 1, 1 directly
    def initialize_four_parameters(self):
        return torch.tensor(0.5), torch.tensor(1), torch.tensor(1), torch.tensor(1)

    # Whether the EM algorithm can stop
    def stop_satisfy(self, last_parameters, new_parameters, loop_number):
        changed_values = np.abs(np.array(new_parameters) - np.array(last_parameters))

        if loop_number >= 100 or (changed_values.max() < 1e-3):
            return True
        else:
            return False

    def train_calibration(self, scores):
        lr = 0.01
        Loop_alpha, Loop_mu, Loop_sigma, Loop_lambda = self.initialize_four_parameters()
        loop_number = 0
        while (True):
            # E-step: Fix the parameters and get the probabilities can minimize the LL function
            current_target = self.calculate_pro_equ15(scores, Loop_alpha, Loop_mu, Loop_sigma, Loop_lambda)
            # M-step: Fix the probabilities and get the four parameters can minimize the LL function

            new_Loop_alpha, new_Loop_mu, new_Loop_sigma, new_Loop_lambda = self.get_minimize_parameters(current_target, scores)
            # print(loop_number, new_Loop_alpha, new_Loop_mu, new_Loop_sigma, new_Loop_lambda)
            # Check stop condition
            if self.stop_satisfy([new_Loop_alpha.cpu().detach(), new_Loop_mu.cpu().detach(), new_Loop_sigma.cpu().detach(), new_Loop_lambda.cpu().detach()],
                            [Loop_alpha.cpu().detach(), Loop_mu.cpu().detach(), Loop_sigma.cpu().detach(), Loop_lambda.cpu().detach()], loop_number):
                break
            else:
                Loop_alpha = Loop_alpha * (1 - lr) + new_Loop_alpha * lr
                Loop_mu = Loop_mu * (1 - lr) + new_Loop_mu * lr
                Loop_sigma = Loop_sigma * (1 - lr) + new_Loop_sigma * lr
                Loop_lambda = Loop_lambda * (1 - lr) + new_Loop_lambda * lr
                loop_number += 1

        self.alpha_value = Loop_alpha
        self.mu_value = Loop_mu
        self.sigma_value = Loop_sigma
        self.lambda_value = Loop_lambda
        # print('End training!')

    # 输出为 **正常** 的预测概率（未校正）
    def predict(self, feats):
        logits = self(feats)
        output = F.softmax(logits, dim=1)
        return output[:, 0]

    def predict_proba(self, feats):
        if self.alpha_value == None or self.mu_value == None or self.sigma_value == None or self.lambda_value == None:
            print("Cannot directly compute without alpha mu sigma and lambda parameters")
            exit(1)
        else:
            logits = self(feats)
            output = F.softmax(logits, dim=1)
            scores = output[:, 0]

            return self.calculate_pro_equ15(scores, self.alpha_value, self.mu_value, self.sigma_value, self.lambda_value)