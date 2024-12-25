# loss function
# from func.structure_acc import cal_structure_acc, get_clusters_and_centers_of_slices

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np


def dice_loss_power_weights_for_learningloss(
    pred, target, weights, alpha=0.5, delta=0.000001
):
    """
    pred: tensor with shape [batch_size, C, D, H, W] or similar
    target: tensor with shape [batch_size, D, H, W] or similar
    weights: tensor with shape [batch_size, D, H, W] or similar
    """
    smooth = 1
    delta = 0.1

    # Flatten the spatial dimensions but keep the batch dimension
    iflat = pred.view(pred.size(0), -1)  # Shape: [batch_size, -1]
    tflat = target.view(target.size(0), -1)  # Shape: [batch_size, -1]
    weights_flat = weights.view(weights.size(0), -1)  # Shape: [batch_size, -1]

    # Compute the intersection, A_sum, and B_sum for each sample in the batch
    intersection = 2.0 * torch.sum(
        torch.mul(torch.mul(torch.pow(iflat + delta, alpha), tflat), weights_flat),
        dim=1,
    )

    A_sum = torch.sum(
        torch.mul(
            torch.mul(torch.pow(iflat + delta, alpha), torch.pow(iflat + delta, alpha)),
            weights_flat,
        ),
        dim=1,
    )
    B_sum = torch.sum(torch.mul(torch.mul(tflat, tflat), weights_flat), dim=1)

    # Compute the Dice loss for each sample in the batch
    dice_loss = 1 - (intersection + smooth) / (A_sum + B_sum + smooth)

    return dice_loss  # Shape: [batch_size]


def dice_loss_weights_for_learningloss(pred, target, weights):
    """
    pred: tensor with shape [batch_size, C, D, H, W] or similar
    target: tensor with shape [batch_size, D, H, W] or similar
    weights: tensor with shape [batch_size, D, H, W] or similar
    """
    smooth = 0.01

    # Flatten the spatial dimensions but keep the batch dimension
    iflat = pred.view(pred.size(0), -1)  # Shape: [batch_size, -1]
    tflat = target.view(target.size(0), -1)  # Shape: [batch_size, -1]
    weights_flat = weights.view(weights.size(0), -1)  # Shape: [batch_size, -1]

    # Compute the intersection, A_sum, and B_sum for each sample in the batch
    intersection = 2.0 * torch.sum(torch.mul(iflat, tflat) * weights_flat, dim=1)
    A_sum = torch.sum(torch.mul(iflat, iflat) * weights_flat, dim=1)
    B_sum = torch.sum(torch.mul(tflat, tflat) * weights_flat, dim=1)

    # Compute the Dice loss for each sample in the batch
    dice_loss = 1 - (intersection + smooth) / (A_sum + B_sum + smooth)

    return dice_loss  # Shape: [batch_size]


def dice_loss_weights(pred, target, weights):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 0.01

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    weights_flat=weights.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(torch.mul(iflat, tflat), weights_flat))

    #A_sum = torch.sum(torch.mul(torch.mul(iflat, iflat), weights_flat))
    #B_sum = torch.sum(torch.mul(torch.mul(tflat, tflat), weights_flat))
    A_sum = torch.sum(torch.mul(iflat, iflat))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(iflat, tflat))

    A_sum = torch.sum(torch.mul(iflat, iflat))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss_power_weights(pred, target, weights, alpha=0.5, delta=0.000001):
    smooth = 1
    delta = 0.1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    weights_flat=weights.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(torch.mul(torch.pow(iflat+delta, alpha), tflat),weights_flat))

    A_sum = torch.sum(torch.mul(torch.mul(torch.pow(iflat+delta, alpha), torch.pow(iflat+delta, alpha)),weights_flat))
    B_sum = torch.sum(torch.mul(torch.mul(tflat, tflat),weights_flat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_loss_power(pred, target, alpha=0.5, delta=0.000001):
    smooth = 1
    delta = 0.1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(torch.pow(iflat+delta, alpha), tflat))

    A_sum = torch.sum(torch.mul(torch.pow(iflat+delta, alpha), torch.pow(iflat+delta, alpha)))
    B_sum = torch.sum(torch.mul(tflat, tflat))
        
    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))

def dice_accuracy(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    
    intersection = 2. * torch.sum(torch.mul(iflat, tflat))

    A_sum = torch.sum(torch.mul(iflat, iflat))
    B_sum = torch.sum(torch.mul(tflat, tflat))
    
    return (intersection) / (A_sum + B_sum + 0.0001)
