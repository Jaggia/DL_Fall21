
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py

def one_hot(index, classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    index = index.to(device)
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0).to(device)
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1)).to(device)
        mask = Variable(mask, volatile=index.volatile).to(device)

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7, weight= None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.weight = weight

    def forward(self, input, target):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input = input.to(device)
        target = target.to(device)
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1).to(device)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        if self.weight == None:
            loss = loss * (1 - logit) ** self.gamma
        else:
            loss = loss * (1 - logit) ** self.gamma * self.weight # focal loss

        return loss.sum()
