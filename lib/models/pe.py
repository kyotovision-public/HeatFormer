# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
# borrowed from U-HMR (https://github.com/XiaobenLi00/U-HMR)
# ------------------------------------------------------------------------

import math

import torch
import torch.nn as nn
# from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING
# from mmcv.runner import BaseModule

# @POSITIONAL_ENCODING.register_module()
class SinePositionalEncoding3D(nn.Module):
    """Position encoding with sine and cosine functions.
    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 context_dim,
                 temperature=10000,
                 normalize=False,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.):
        super(SinePositionalEncoding3D, self).__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feats = context_dim // 2
        self.context_dim = context_dim
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset
        self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.context_dim*3//2, self.context_dim*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.context_dim*4, self.context_dim, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, x):
        """Forward function for `SinePositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, N, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        # mask = mask.to(torch.int)
        # not_mask = 1 - mask  # logical_not
        not_mask = torch.ones(x.shape[0], x.shape[1], x.shape[3], x.shape[4], dtype=torch.bool, device=x.device)
        n_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            n_embed = (n_embed + self.offset) / \
                      (n_embed[:, -1:, :, :] + self.eps) * self.scale
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, :, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=not_mask.device)
        # dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        dim_t = self.temperature**(2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_feats)
        pos_n = n_embed[:, :, :, :, None] / dim_t
        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, N, H, W = not_mask.size()
        pos_n = torch.stack(
            (pos_n[:, :, :, :, 0::2].sin(), pos_n[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)
        pos_x = torch.stack(
            (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()),
            dim=4).view(B, N, H, W, -1)
        pos = torch.cat((pos_n, pos_y, pos_x), dim=4).permute(0, 1, 4, 2, 3)

        pos = self.adapt_pos3d(pos.flatten(0, 1)).view(x.size())

        return pos
    
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, 
                 context_dim,
                 temperature=10000,
                 normalize=False,
                 scale=2*math.pi,
                 esp=1e-6,
                 offset=0.):
        super().__init__()
        self.num_feats = context_dim // 2
        self.context_dim = context_dim
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.esp = esp
        self.offset = offset

    def forward(self, x):
        not_mask = torch.ones(x.shape[0], x.shape[2], x.shape[3], device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=not_mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos