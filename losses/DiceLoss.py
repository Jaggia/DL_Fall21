import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# tensor([[ 0.0461, -0.0012, -0.0089,  ..., -0.0297, -0.0044,  0.0658],
#         [ 0.0568, -0.0041, -0.0287,  ..., -0.0132, -0.0036,  0.0446],
#         [ 0.0461,  0.0206, -0.0017,  ..., -0.0199,  0.0014,  0.0403],
#         ...,
#         [ 0.0769, -0.0164, -0.0437,  ..., -0.0043,  0.0380,  0.0038],
#         [ 0.0532, -0.0040, -0.0012,  ..., -0.0229,  0.0097,  0.0503],
#         [ 0.0660,  0.0065,  0.0043,  ..., -0.0303,  0.0268,  0.0512]],
#        device='cuda:0', grad_fn=<SqueezeBackward1>)
# tensor([ 8,  1, 15,  6,  4, 28, 14, 13,  1, 28, 39, 11, 19,  5,  4, 14,  2, 12,
#          8,  8,  1, 10, 36, 14,  3,  2,  0,  6,  3, 18,  1, 10, 17,  1, 13, 30,
#          4,  7, 24, 28, 34,  0,  0, 25,  5,  0,  2,  6, 28,  3, 28, 20,  0,  1,
#         19,  6, 26,  4,  1,  6,  5, 19,  2,  4], device='cuda:0')
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, inputs, targets, smooth=1):
        # print(inputs)
        # print(inputs.shape)  # torch.Size([64, 41])
        # print(targets)
        # print(targets.shape)  # torch.Size([64])

        # comment out for diff implementations
        # return self.kornia(inputs, targets)
        return self.dice_loss(targets, inputs)

    def class_avged(self, inputs, targets, smooth):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        # flatten label and prediction tensors
        inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        targets = targets.view(targets.shape[0], targets.shape[1], -1)
        # intersection = (inputs * targets).sum()
        intersection = (inputs * targets).sum(0).sum(1)
        # dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        dice = (2. * intersection + smooth) / (inputs.sum(0).sum(1) + targets.sum(0).sum(1) + smooth)

        return 1 - dice.mean()

    def kornia(self, input, target):
        # https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/dice.html
        """
        Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
        """
        input = input.unsqueeze(-1).unsqueeze(-1)
        target = target.unsqueeze(-1).unsqueeze(-1)
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = F.one_hot(target, num_classes=input.shape[1])

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)

    def dice_loss(self, true, logits, eps=1e-7):
        # https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
        """
        Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        # print(true.shape)
        # print(logits.shape)
        true = true.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        logits = logits.unsqueeze(-1).unsqueeze(-1)
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)
