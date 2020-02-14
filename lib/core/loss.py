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

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        # self.criterion = nn.MSELoss(size_average=True)
        self.criterion = nn.SmoothL1Loss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_vis, log_var):
        log_var = log_var * target_vis.squeeze(-1)
        loss_coord = torch.abs(output - target) * target_vis
        loss_coord = (loss_coord[:, :, 0] + loss_coord[:, :, 1]) / 2.
        loss_all = torch.exp(-log_var) * loss_coord + 1*log_var
        return loss_coord.mean(), loss_all.mean()


    # def forward(self, output, target, target_weight):
    #     batch_size = output.size(0)
    #     num_joints = output.size(1)
    #     heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
    #     heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
    #     loss = 0
    #
    #     for idx in range(num_joints):
    #         heatmap_pred = heatmaps_pred[idx].squeeze()
    #         heatmap_gt = heatmaps_gt[idx].squeeze()
    #         if self.use_target_weight:
    #             loss += 0.5 * self.criterion(
    #                 heatmap_pred.mul(target_weight[:, idx]),
    #                 heatmap_gt.mul(target_weight[:, idx])
    #             )
    #         else:
    #             loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
    #
    #     return loss / num_joints

