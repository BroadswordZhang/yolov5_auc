#coding=utf-8
import torch
import torch.nn.functional as F

from torch import nn


def one_hot(y, num_classes,device):
    '''Return (batch_size x num_classes) shaped 
    one hot torch tensor'''
    
    y_ = torch.zeros((y.shape[0], num_classes)).to(device, dtype=torch.long)
    y_[torch.arange(y.shape[0]), y] = 1
    return y_


class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, alpha = 0.5, num_classes=2, device="cpu",reduction = "mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes= num_classes
        self.reduction = reduction
        self.device = device
    
    def forward(self, inputs, targets):
        """
        cal culates loss
        logits: batch_size * labels_length * seq_length
        labels: batch_size * seq_length
        """
        p = F.softmax(inputs, dim = 1) #(batch_size, n_class)
        log_p = F.log_softmax(inputs, dim = 1) #(batch_size, n_class)

        ce_loss = F.nll_loss(input = log_p, target = targets, 
            ignore_index = 8, reduction = self.reduction) #(batch_size,)

        targets_ = one_hot(targets, self.num_classes, self.device) # (batch_size,n_class)
        p_t = torch.sum(targets_ * p, dim = 1) #(batch_size,)

        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean() #(1)
        if self.reduction == 'sum':
            focal_loss = focal_loss.sum() #(1)
        if self.reduction == 'none':
            pass
        
        return focal_loss



