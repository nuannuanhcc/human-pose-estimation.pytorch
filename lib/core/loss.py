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
from torch.nn import functional as F
from core.config import config

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        # self.criterion = nn.MSELoss(size_average=True)
        self.criterion = nn.SmoothL1Loss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target_hm, target, target_vis, scale, loss_norm, lamda=0.1):
        loss = 0
        loss_coord = 0
        loss_hm = 0
        pre = output[0]
        pre_hm = output[1]
        if 'arg_loss' in config.all_loss:
            loss_coord = torch.abs(pre - target) * target_vis
            loss_coord = (loss_coord[:, :, 0] + loss_coord[:, :, 1]) / 2.
            loss_coord = loss_coord.mean()
            loss_norm = (loss_norm * target_vis.squeeze(-1)).mean()
            loss = loss + loss_coord + lamda * loss_norm
        if 'hm_loss' in config.all_loss:
            batch_size = pre_hm.size(0)
            num_joints = pre_hm.size(1)
            heatmaps_pred = pre_hm.reshape((batch_size, num_joints, -1)).split(1, 1)
            heatmaps_gt = target_hm.reshape((batch_size, num_joints, -1)).split(1, 1)
            for idx in range(num_joints):
                heatmap_pred = heatmaps_pred[idx].squeeze()
                heatmap_gt = heatmaps_gt[idx].squeeze()
                if self.use_target_weight:
                    loss_hm += self.criterion(
                        heatmap_pred.mul(target_vis[:, idx]),
                        heatmap_gt.mul(target_vis[:, idx])
                    )
                else:
                    loss_hm += self.criterion(heatmap_pred, heatmap_gt)
            loss_hm = loss_hm.mean()
            loss += loss_hm
        return loss, loss_coord, loss_hm, loss_norm

