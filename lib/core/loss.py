from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.geometry import batch_rodrigues

class CoordLoss_2d(nn.Module):
    def __init__(self, batch, view_sum, loss_type='l1', has_valid=False):
        super(CoordLoss_2d, self).__init__()

        self.has_valid = has_valid
        if loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss(reduction='none')
        
        self.batch = batch
        self.view_sum = view_sum

    def forward(self, pred, target, target_valid=None):
        if self.has_valid:
            pred, target = pred * target_valid, target * target_valid

        loss = self.criterion(pred, target).sum(dim=-1).sum(dim=-1)

        if self.view_sum:
            loss = loss.reshape(self.batch, -1)
            loss.sum()

        return loss.mean()

class CoordLoss_3d(nn.Module):
    def __init__(self, batch, view_sum, loss_type='l1', has_valid=False):
        super(CoordLoss_3d, self).__init__()

        self.has_valid = has_valid
        if loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.criterion = nn.MSELoss(reduction='none')

        self.batch = batch
        self.view_sum = view_sum

    def forward(self, pred, target, target_valid=None):
        if self.has_valid:
            pred, target = pred * target_valid, target * target_valid

        pred = pred - pred[:, 0, None]
        target = target - target[:, 0, None]

        loss = self.criterion(pred, target).sum(dim=-1).sum(dim=-1)

        if self.view_sum:
            loss = loss.reshape(self.batch, -1)
            loss.sum()

        return loss.mean()

class GlobalOrientLoss(nn.Module):
    def __init__(self, batch, view_sum) -> None:
        super(GlobalOrientLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        
        self.batch = batch
        self.view_sum = view_sum
    
    def forward(self, pred, target):
        loss = self.criterion(pred, target).sum(dim=-1).sum(dim=-1)

        if self.view_sum:
            loss = loss.reshape(self.batch, -1)
            loss.sum()

        return loss.mean()
    
class PoseLoss(nn.Module):
    def __init__(self, batch, view_sum):
        super(PoseLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

        self.batch = batch
        self.view_sum = view_sum
    
    def forward(self, pred, target):
        loss = self.criterion(pred, target).sum(dim=-1).sum(dim=-1)

        if self.view_sum:
            loss = loss.reshape(self.batch, -1)
            loss.sum()

        return loss.mean()

class ShapeLoss(nn.Module):
    def __init__(self, batch, view_sum):
        super(ShapeLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

        self.batch = batch
        self.view_sum = view_sum
    
    def forward(self, pred, target):
        loss = self.criterion(pred, target).sum(dim=-1)

        if self.view_sum:
            loss = loss.reshape(self.batch, -1)
            loss.sum()

        return loss.mean()
    
class NormalVectorLoss(nn.Module):
    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt):
        face = torch.LongTensor(self.face).cuda()

        v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
        v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
        v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
        v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))
        loss = torch.cat((cos1, cos2, cos3), 1)
        return loss.mean()

class HeatmapLoss(nn.Module):
    def __init__(self, batch, view_sum=False):
        super().__init__()
        self.view_sum = view_sum
        self.criterion = nn.MSELoss(reduction='none')
    
    def forward(self, heatmap_pred, heatmap_smpl):
        loss = self.criterion(heatmap_pred, heatmap_smpl).sum(dim=-1).sum(dim=-1).sum(dim=-1)

        if self.view_sum:
            loss = loss.reshape(self.batch, -1)
            loss.sum()

        return loss.mean()

class Joint2dSmoothLoss(nn.Module):
    def __init__(self):
        super(Joint2dSmoothLoss, self).__init__()
        factor = torch.as_tensor(8.0)
        alpha = torch.as_tensor(-10.0)
        self.register_buffer('factor', factor)
        self.register_buffer('alpha', alpha)

    def forward(self, joint_2d, gt, target_weight=None):
        """

        :param joint_2d: (batch*nview, njoint, 2)
        :param gt:
        :param target_weight: (batch*nview, njoint, 1)
        :return:
        """
        x = torch.sum(torch.abs(joint_2d - gt), dim=2)  # (batch*nview, njoint)
        x_scaled = ((x / self.factor) ** 2 / torch.abs(self.alpha-2) + 1) ** (self.alpha * 0.5) -1
        x_final = (torch.abs(self.alpha) - 2) / self.alpha * x_scaled

        loss = x_final
        if target_weight is not None:
            cond = torch.squeeze(target_weight) < 0.5
            loss = torch.where(cond, torch.zeros_like(loss), loss)
        loss_mean = loss.mean()
        return loss_mean * 1000.0


def get_loss(batch, face, view_sum=False):
    loss = CoordLoss_3d(batch, view_sum=view_sum), \
            CoordLoss_2d(batch, view_sum=view_sum), \
            GlobalOrientLoss(batch, view_sum=view_sum), \
            PoseLoss(batch, view_sum=view_sum), \
            ShapeLoss(batch, view_sum=view_sum), \
            CoordLoss_3d(batch, view_sum=view_sum), \
            NormalVectorLoss(face), \
            HeatmapLoss(batch, view_sum=view_sum)

    return loss