# loss function
# from func.structure_acc import cal_structure_acc, get_clusters_and_centers_of_slices


import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
def equalized_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_b=2,
    scale_factor=8,
    reduction="mean",
):
    """EFL loss"""
    ce_loss = F.cross_entropy(logits, targets, reduction="none")
    outputs = F.cross_entropy(logits, targets)  # 求导使用，不能带 reduction 参数
    log_pt = -ce_loss
    pt = torch.exp(log_pt)  # softmax 函数打分

    targets = targets.view(-1, 1)  # 多加一个维度，为使用 gather 函数做准备
    grad_i = torch.autograd.grad(outputs=-outputs, inputs=logits)[0]  # 求导
    grad_i = grad_i.gather(1, targets)  # 每个类对应的梯度
    pos_grad_i = F.relu(grad_i).sum()
    neg_grad_i = F.relu(-grad_i).sum()
    neg_grad_i += 1e-9  # 防止除数为0
    grad_i = pos_grad_i / neg_grad_i
    grad_i = torch.clamp(grad_i, min=0, max=1)  # 裁剪梯度

    dy_gamma = gamma_b + scale_factor * (1 - grad_i)
    dy_gamma = dy_gamma.view(-1)  # 去掉多的一个维度
    # weighting factor
    wf = dy_gamma / gamma_b
    weights = wf * (1 - pt) ** dy_gamma

    efl = weights * ce_loss

    if reduction == "sum":
        efl = efl.sum()
    elif reduction == "mean":
        efl = efl.mean()
    else:
        raise ValueError(f"reduction '{reduction}' is not valid")
    return efl


def balanced_equalized_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha_t=0.25,
    gamma_b=2,
    scale_factor=8,
    reduction="mean",
):
    """balanced EFL loss"""
    return alpha_t * equalized_focal_loss(
        logits, targets, gamma_b, scale_factor, reduction
    )


def focal_loss(input, target, gamma=0, alpha=None, size_average=True):
    """
    计算 Focal Loss，用于分类任务的损失计算。

    :param input: 预测的 logits (未经过 softmax)，形状为 (N, C) 或 (N, C, H, W)。
    :param target: 真实标签，形状为 (N,) 或 (N, H, W)。
    :param gamma: 聚焦参数 gamma，默认为 0。
    :param alpha: 类别权重平衡参数。如果是 float 或 int，则表示类别 0 和 1 的权重 [alpha, 1-alpha]；
                  如果是 list，则表示所有类别的权重。
    :param size_average: 如果为 True，则对 batch 中的损失取均值；如果为 False，则对损失求和。
    :return: 计算后的 Focal Loss。
    """
    # 如果输入的维度大于 2，需要进行展平处理
    if input.dim() > 2:
        # 调整形状为 (N, C, H*W)
        input = input.view(input.size(0), input.size(1), -1)
        # 调换通道和像素维度顺序，变为 (N, H*W, C)
        input = input.transpose(1, 2)
        # 再展平为二维张量 (N*H*W, C)
        input = input.contiguous().view(-1, input.size(2))
    # 将目标标签展平为一列
    target = target.view(-1, 1)

    # 计算 log-softmax，得到每个类别的对数概率
    logpt = F.log_softmax(input, dim=1)
    # 根据 target 索引选取目标类别的对数概率
    logpt = logpt.gather(1, target)
    # 展平 logpt
    logpt = logpt.view(-1)
    # 计算 pt，即概率值
    pt = logpt.exp()

    # 如果 alpha 不为空，应用类别权重
    if alpha is not None:
        if isinstance(alpha, (float, int)):
            # 如果 alpha 是标量，将其转换为张量形式 [alpha, 1-alpha]
            alpha = torch.Tensor([alpha, 1 - alpha]).to(input.device)
        elif isinstance(alpha, list):
            # 如果 alpha 是列表，直接转换为张量
            alpha = torch.Tensor(alpha).to(input.device)
        # 根据 target 索引获取对应类别的权重
        at = alpha.gather(0, target.view(-1).cpu().long())
        # 将权重应用到 logpt
        logpt = logpt * at

    # 根据 Focal Loss 的公式计算损失
    loss = -1 * (1 - pt) ** gamma * logpt
    # 如果需要取平均，返回均值；否则返回损失和
    return loss.mean() if size_average else loss.sum()


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
