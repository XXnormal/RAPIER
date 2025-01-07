import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LSTM_AE_GMM(nn.Module):
    def __init__(self, emb_dim, input_size, hidden_size, dropout, max_len,
        est_hidden_size, est_output_size, device=0, est_dropout=0.5,
        learning_reat=0.0001, lambda1=0.1, lambda2=0.0001):
        super(LSTM_AE_GMM, self).__init__()

        self.max_len = max_len # 最大包长度（已提前处理）
        self.emb_dim = emb_dim # 词向量维度
        self.input_size = input_size # 输入包长度序列长度（AE最终预测长度）
        self.hidden_size = hidden_size # GRU输出维度
        self.dropout = dropout 
        self.device = device

        torch.cuda.set_device(self.device)
        
        # 词向量层
        self.embedder = nn.Embedding(self.max_len, self.emb_dim)
        # GRU Encoder层
        self.encoders = nn.ModuleList([
            nn.GRU(
                input_size=self.emb_dim,
                hidden_size=self.hidden_size,
                batch_first=True,
                bidirectional=True,
            ).cuda(), 
            nn.GRU(
                input_size=self.hidden_size * 2,
                hidden_size=self.hidden_size,
                batch_first=True,
                bidirectional=True,
            ).cuda()
        ])
        # GRU Decoder层 （按照FS-Net图所示）
        self.decoders = nn.ModuleList([
            nn.GRU(
                input_size=self.hidden_size * 4,
                hidden_size=self.hidden_size,
                batch_first=True,
                bidirectional=True,
            ).cuda(), 
            nn.GRU(
                input_size=self.hidden_size * 2,
                hidden_size=self.hidden_size,
                batch_first=True,
                bidirectional=True,
            ).cuda()
        ])
        # Reconstruct层
        self.rec_fc1 = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.rec_fc2 = nn.Linear(self.hidden_size, self.max_len)
        self.rec_softmax = nn.Softmax(dim=2)
        # 计算Reconstruct的loss值
        self.cross_entropy = nn.CrossEntropyLoss()

        # TODO: estimation work
        self.est_hidden_size = est_hidden_size
        self.est_output_size = est_output_size
        self.est_dropout = est_dropout
        self.fc1 = nn.Linear(4 * self.hidden_size, self.est_hidden_size)
        self.fc2 = nn.Linear(self.est_hidden_size, self.est_output_size)
        self.est_drop = nn.Dropout(p=self.est_dropout)
        self.softmax = nn.Softmax(dim=1)

        self.training = False

        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def encode(self, x):
        # encoder层
        torch.cuda.set_device(self.device)
        embed_x = self.embedder(x.long())
        if self.training is True:
            embed_x = F.dropout(embed_x)
        outputs = [embed_x]
        hs = []
        for layer in range(2):
            gru = self.encoders[layer]
            output, h = gru(outputs[-1])
            outputs.append(output)
            hs.append(torch.transpose(h, 0, 1).reshape(
                -1, 2 * self.hidden_size))
        res = torch.cat(outputs[1:], dim=2)
        res_h = torch.cat(hs, dim=1)
        return res_h
    
    def decode_input(self, x):
        torch.cuda.set_device(self.device)
        y = x.reshape(-1, 1, 4 * self.hidden_size)
        y = y.repeat(1, self.input_size, 1)
        return y
    
    def decode(self, x):
        # decoder层
        torch.cuda.set_device(self.device)
        input = x.view(-1, self.input_size, 4 * self.hidden_size)
        outputs = [input]
        hs = []
        for layer in range(2):
            gru = self.decoders[layer]
            output, h = gru(outputs[-1])
            outputs.append(output)
            hs.append(torch.transpose(h, 0, 1).reshape(
                -1, 2 * self.hidden_size))
        res = torch.cat(outputs[1:], dim=2)
        res_h = torch.cat(hs, dim=1)
        return res, res_h
    
    def reconstruct(self, x, y):
        # reconstruct层
        torch.cuda.set_device(self.device)
        x_rec = self.rec_fc2(F.selu(self.rec_fc1(x)))
        loss = F.cross_entropy(x_rec.view(-1, self.max_len), y.long().view(-1), reduction='none')
        loss = loss.view(-1, self.input_size)
        mask = y.bool()
        loss_ret = torch.sum(loss * mask, dim=1) / torch.sum(mask, dim=1)
        return loss_ret
    
    def estimate(self, x):
        torch.cuda.set_device(self.device)
        x = x.view(-1, 4 * self.hidden_size)
        res = self.est_drop(F.tanh(self.fc1(x)))
        res = self.softmax(self.fc2(res))
        return res
    
    def feature(self, input):
        torch.cuda.set_device(self.device)
        res_encode_h = self.encode(input.float())
        return res_encode_h

    def predict(self, input):
        torch.cuda.set_device(self.device)
        res_encode_h = self.encode(input.float())
        decode_input = self.decode_input(res_encode_h)
        res_decode, res_decode_h = self.decode(decode_input)
        loss_all = self.reconstruct(res_decode, input)
        return res_encode_h, loss_all

    def loss(self, input):
        torch.cuda.set_device(self.device)
        _, loss_all = self.predict(input)
        return torch.mean(loss_all, dim=0)
    
    def classify_loss(self, input, labels):
        torch.cuda.set_device(self.device)
        feats, rec_loss = self.predict(input)
        score = self.estimate(feats)
        return F.cross_entropy(score, labels) + torch.mean(rec_loss, dim=0)
    
    def classify_loss_1(self, input, labels):
        torch.cuda.set_device(self.device)
        feats, rec_loss = self.predict(input)
        score = self.estimate(feats)
        return F.cross_entropy(score, labels, reduce = False) + rec_loss

    def train_mode(self):
        self.training = True
    
    def test_mode(self):
        self.training = False

    def to_cpu(self):
        self.device = None
        for encoder in self.encoders:
            encoder = encoder.cpu()
        for decoder in self.decoders:
            decoder = decoder.cpu()
    
    def to_cuda(self, device):
        self.device = device
        torch.cuda.set_device(self.device)
        for encoder in self.encoders:
            encoder = encoder.cuda()
        for decoder in self.decoders:
            decoder = decoder.cuda()
