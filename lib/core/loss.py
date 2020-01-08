# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight, sigma):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        loss_original = 0
        loss_var = 0
        var = torch.ones_like(sigma)
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                pre = heatmap_pred.mul(target_weight[:, idx])
                gt = heatmap_gt.mul(target_weight[:, idx])
                mse = torch.mean((pre - gt) ** 2, dim=1)
                mse = torch.exp(-sigma[:, idx]) * mse + sigma[:, idx]
                loss += 0.5 * torch.mean(mse)
                loss_original += 0.5 * self.criterion(pre, gt)

                var_pre = torch.exp(sigma[:, idx] / 2).mul(target_weight[:, idx])
                var_gt = var[:, idx].mul(target_weight[:, idx])
                loss_var += self.criterion(var_pre, var_gt)
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        return (loss + 10 * loss_var) / num_joints, loss_original / num_joints
