from smplx import SMPL as _SMPL
from lib.core.config import BASE_DATA_DIR, SMPL_MEAN_PARAMS, SMPL_MODEL_DIR
import time
import torch
import numpy as np
import os.path as osp

from einops import rearrange

from lib.utils.geometry import rot6d_to_rotmat, rot6d_to_rotmat_g, aa_to_rotmat, perspective_projection
from lib.models.composer import batch_heatmap_generator, heatmap_generator, affine_transform

class SMPL_orig(_SMPL):
    def __init__(self, J_regressor_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        J_regressor_h36m = torch.from_numpy(np.load(osp.join(BASE_DATA_DIR, J_regressor_path))).float()
        self.register_buffer('J_regressor_h36m', J_regressor_h36m)
        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:]).float().reshape(24, 6) # (24, 6)
        init_shape = torch.from_numpy(mean_params['shape'][:]).float().unsqueeze(0) # (1, 10)
        init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)
    
    def forward(self, *args, **kwargs):
        smpl_out = super().forward(*args, **kwargs)
        batch = smpl_out.vertices.shape[0]
        J_regressor_h36m_batch = self.J_regressor_h36m[None].expand(batch, -1, -1)
        regress_joints = torch.matmul(J_regressor_h36m_batch, smpl_out.vertices).reshape(batch, -1, 3)

        smpl_return = {
            'verts' : smpl_out.vertices,
            'regress_joints' : regress_joints
        }

        return smpl_return

    
    def make_heatmap(
            self, 
            K : torch.Tensor,
            regress_joints : torch.Tensor,
            heatmap_size : int, 
            image_size : int, 
            sigma : int, 
            trans : torch.Tensor
        ):
        """
        regress_joints : joints regressed from SMPL mesh on Img coords (batch n_view n_joint, 2)
        heatmap_size : size of heatmap                                 (int, )
        trans : rot and scale from original image                      (batch n_view 2, 3)
        """

        device = regress_joints.device
        batch, n_view = trans.shape[:2]
        regress_joints = rearrange(regress_joints, '(b v) j c -> b v j c', b=batch)
        n_joints = regress_joints.shape[2]
        # 3d => 2d
        reg_3d = regress_joints / regress_joints[..., -1:]
        reg_2d = torch.einsum('bvij, bvkj->bvki', K, reg_3d)[..., :2]
        reg_joints = torch.cat((reg_2d, torch.ones(batch, n_view, n_joints, 1).to(device)), dim=-1)
        assert reg_joints.shape == (batch, n_view, n_joints, 3)

        # original -> img crop
        reg_joints_trans = affine_transform(reg_joints, trans)
        joints_vis = torch.ones(batch, n_view, n_joints)
        out_of_range_idx = torch.where((reg_joints_trans>=image_size) | (reg_joints_trans<0))
        joints_vis[out_of_range_idx[:-1]] = 0
        
        heatmap = batch_heatmap_generator(reg_joints_trans, joints_vis, heatmap_size, image_size, sigma)
        aft_joints = reg_joints_trans

        return heatmap
    
    def make_smpl(self, K, joints_3d_cam, trans, delta_global_orient, delta_cam):
        batch_size, n_view, n_joints = joints_3d_cam.shape[:3]
        pose6d = self.init_pose.unsqueeze(0).repeat(batch_size * n_view, 1, 1).reshape(batch_size * n_view, -1)
        if delta_global_orient is not None:
            pose6d[:, :6] = pose6d[:, :6] + delta_global_orient
        global_orient_mat = rot6d_to_rotmat_g(pose6d[:, :6]).reshape(-1, 1, 3, 3)
        body_pose_mat = rot6d_to_rotmat(pose6d[:, 6:]).reshape(-1, 23, 3, 3)
        pose = torch.cat([global_orient_mat, body_pose_mat], dim=1)
        betas = self.init_shape.repeat(batch_size * n_view, 1)                 # (batch * n_view 10)
        cam = self.init_cam.expand(batch_size * n_view, -1)
        if delta_cam is not None:
            cam = cam + delta_cam

        pose = pose.reshape(batch_size * n_view, 24, 3, 3)

        smpl_out = self.forward(
            global_orient=pose[:, 0].reshape(-1, 1, 3, 3),
            body_pose=pose[:, 1:].reshape(-1, 23, 3, 3),
            betas=betas,
            pose2rot=False
        )

        reg = smpl_out['regress_joints'].reshape(batch_size, n_view, -1, 3) # (batch n_view 17 3)

        transl = joints_3d_cam[:, :, 0] - reg[:, :, 0]
        reg = reg + transl[:, :, None]

        reg = reg / reg[..., -1:]
        reg_2d = torch.einsum('bvij,bvkj->bvki', K, reg)

        reg_trans = affine_transform(reg_2d, trans)

        return reg_trans, pose6d, betas, cam, None
        
    
    def make_init_heatmap(
            self,
            K : torch.Tensor,
            joints_3d_cam : torch.Tensor,
            image_size : int,
            heatmap_size : int,
            sigma : int,
            trans : torch.Tensor,
            delta_global_orient : torch.Tensor,
            delta_cam : torch.Tensor,
            eval=False, # for visualize
            g_rot=False,
            g_rot_all=False
        ):
        batch_size, n_view, n_joints = joints_3d_cam.shape[:3]
        pose6d = self.init_pose.unsqueeze(0).repeat(batch_size * n_view, 1, 1).reshape(batch_size * n_view, -1)
        if delta_global_orient is not None:
            pose6d[:, :6] = pose6d[:, :6] + delta_global_orient
        if g_rot:
            global_orient_mat = rot6d_to_rotmat_g(pose6d[:, :6]).reshape(-1, 1, 3, 3)
            body_pose_mat = rot6d_to_rotmat(pose6d[:, 6:]).reshape(-1, 23, 3, 3)
            pose = torch.cat([global_orient_mat, body_pose_mat], dim=1)
        elif g_rot_all:
            pose = rot6d_to_rotmat_g(pose6d).reshape(-1, 24, 3, 3) 
        else:
            pose = rot6d_to_rotmat(pose6d).reshape(-1, 24, 3, 3)                   # (batch * n_view * 24 3 3)
        betas = self.init_shape.repeat(batch_size * n_view, 1)                 # (batch * n_view 10)
        cam = self.init_cam.expand(batch_size * n_view, -1)
        if delta_cam is not None:
            cam = cam + delta_cam

        pose = pose.reshape(batch_size * n_view, 24, 3, 3)

        # Calculate SMPL
        smpl_out = self.forward(
            global_orient=pose[:, 0].reshape(-1, 1, 3, 3),
            body_pose=pose[:, 1:].reshape(-1, 23, 3, 3),
            betas=betas,
            pose2rot=False
        )

        reg = smpl_out['regress_joints'].reshape(batch_size, n_view, -1, 3) # (batch n_view 17 3)

        # aligned position
        transl = joints_3d_cam[:, :, 0] - reg[:, :, 0]
        reg = reg + transl[:, :, None]

        # reprojection
        reg = reg / reg[..., -1:]
        reg_2d = torch.einsum('bvij,bvkj->bvki', K, reg)

        # transform
        reg_trans = affine_transform(reg_2d, trans)
        joints_vis = torch.ones(batch_size, n_view, n_joints)
        out_of_range_idx = torch.where((reg_trans>=image_size) | (reg_trans<0))
        joints_vis[out_of_range_idx[:-1]] = 0

        heatmap = batch_heatmap_generator(reg_trans, joints_vis, heatmap_size, image_size, sigma)
        aft_joints = reg_trans

        if not eval:
            return heatmap, pose6d, betas, cam, None
        else:
            return heatmap, pose6d, betas, cam, smpl_out
        
    def make_heatmap_from2d(
            self, 
            batch_size,
            j2d_norm : torch.Tensor, 
            image_size : int,
            image_size_ad : int, 
            heatmap_size : int, 
            sigma : int, 
            delta_global_orient : torch.Tensor = None,
            delta_cam : torch.Tensor = None,
            focal_length : int = 0,
            eval : bool = False,
            g_rot=False
        ):

        if j2d_norm is None:
            batch_view = delta_global_orient.shape[0]
            pose6d = self.init_pose.unsqueeze(0).repeat(batch_view, 1, 1).reshape(batch_view, -1)
            if delta_global_orient is not None:
                pose6d[:, :6] = pose6d[:, :6] + delta_global_orient
            if g_rot:
                global_orient_mat = rot6d_to_rotmat_g(pose6d[:, :6]).reshape(-1, 1, 3, 3)
                body_pose_mat = rot6d_to_rotmat(pose6d[:, 6:]).reshape(-1, 23, 3, 3)
                pose = torch.cat([global_orient_mat, body_pose_mat], dim=1)
            else:
                pose = rot6d_to_rotmat(pose6d).reshape(-1, 24, 3, 3)                   # (batch * n_view * 24 3 3)
            betas = self.init_shape.repeat(batch_view, 1)                 # (batch * n_view 10)
            cam = self.init_cam.expand(batch_view, -1)
            if delta_cam is not None:
                cam = cam + delta_cam

            smpl_out = self.forward(
                global_orient=pose[:, 0].reshape(-1, 1, 3, 3),
                body_pose=pose[:, 1:].reshape(-1, 23, 3, 3),
                betas=betas,
                pose2rot=False
            )

            focal_length_batch = focal_length * torch.ones(cam.shape[0], 2).float().to(cam.device)
            pred_cam_t = torch.stack([cam[:, 1], cam[:, 2], 2 * focal_length_batch[:, 0] / (image_size * cam[:, 0] + 1e-9)], dim=-1)

            j2d_norm = perspective_projection(
                points=smpl_out['regress_joints'],
                translation=pred_cam_t,
                focal_length=focal_length_batch / image_size
            )

            j2d_norm = j2d_norm.reshape(batch_size, -1, *j2d_norm.shape[1:])

        j2d = (j2d_norm + 0.5) * image_size_ad

        batch_size, n_view, n_joints = j2d.shape[:3]
        joints_vis = torch.ones(batch_size, n_view, n_joints)
        out_of_range_idx = torch.where((j2d>=image_size_ad) | (j2d<0))
        joints_vis[out_of_range_idx[:-1]] = 0

        heatmap = batch_heatmap_generator(j2d, joints_vis, heatmap_size, image_size_ad, sigma)

        if delta_global_orient is not None:
            if not eval:
                return heatmap, pose6d, betas, cam, None
            else:
                return heatmap, pose6d, betas, cam, smpl_out
        else:
            return heatmap


def get_smpl_faces():
    smpl = _SMPL(
        SMPL_MODEL_DIR,
        batch_size=1,
        create_transl=False
    )
    return torch.from_numpy(smpl.faces.astype(np.int32)).long()