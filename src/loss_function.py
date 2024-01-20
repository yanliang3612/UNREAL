import torch
import torch.nn.functional as F
from torch import autograd
from torch import nn


class My_loss:

    def __init__(self, input, target,round,data,running_train_mask,args):
        self.input=input
        self.target=target
        self.round=round
        self.data=data
        self.running_train_mask = running_train_mask
        self.args = args
        self.stride =args.stride



    def weight(self):
        confidence= torch.max(F.softmax(self.input, dim=1), dim=1)[0]
        round_weight = torch.sigmoid(torch.tensor(self.round * self.stride)).to(self.args.device)
        train_confidence=confidence*round_weight
        running_train_index=torch.nonzero(self.running_train_mask)
        data_train_index = torch.nonzero(self.data.train_mask)
        train_confidence[data_train_index] = 1
        weight=train_confidence[self.running_train_mask]
        return weight


    def loss(self):
        weight =self.weight()
        input_true = self.input[self.running_train_mask]
        target_true = self.target[self.running_train_mask]
        pred = log_softmax(input_true)
        put=-pred[range(target_true.shape[0]), target_true]
        loss=(put*weight).mean()
        return loss




class My_end_loss:

    def __init__(self, input, target,data,running_train_mask,args):
        self.input=input
        self.target=target
        self.data=data
        self.running_train_mask = running_train_mask
        self.args = args
        self.stride =args.stride



    def weight(self):
        confidence= torch.max(F.softmax(self.input, dim=1), dim=1)[0]
        train_confidence=confidence
        running_train_index=torch.nonzero(self.running_train_mask)
        data_train_index = torch.nonzero(self.data.train_mask)
        train_confidence[data_train_index] = 1
        weight=train_confidence[self.running_train_mask]
        return weight


    def loss(self):
        weight =self.weight()
        input_true = self.input[self.running_train_mask]
        target_true = self.target[self.running_train_mask]
        pred = log_softmax(input_true)
        put=-pred[range(target_true.shape[0]), target_true]
        loss=(put*weight).mean()
        return loss




def log_softmax(x): return x - x.exp().sum(-1).log().unsqueeze(-1)

