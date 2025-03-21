import torch
import torch.nn as nn
import torchvision.transforms.functional as F

from einops import rearrange

from lib.models.Transformer import TransformerDecoder
from lib.models.pe import SinePositionalEncoding3D, PositionEmbeddingSine
from lib.models.smpl import SMPL_orig, SMPL_MODEL_DIR
from lib.models.composer import batch_heatmap_generator_vis
from lib.utils.geometry import rot6d_to_rotmat, rot6d_to_rotmat_g, perspective_projection, matrix_to_axis_angle, aa_to_rotmat

class HeatFormerEncoderDecoder(nn.Module):
    def __init__(self, cfg, fuser, fuse_encoder, img_backbone, GCestimator, criterion, query_type=1):
        super().__init__()

        npose = 23 * 6

        self.cfg = cfg
        self.fuser = fuser
        self.fuse_encoder = fuse_encoder
        self.img_backbone = img_backbone
        self.GCEstimator = GCestimator

        self.criterion = criterion

        self.query_type = query_type

        if cfg.MODEL.WITH_IMG:
            context_dim = 1024 + 1280
        else:
            context_dim = 1024

        if query_type == 1:
            num_tokens = 1
        elif query_type == 2:
            num_tokens = 16 * 12
        elif query_type == 3:
            num_tokens = 16 * 12 * 4

        transformer_decoder_args = dict(
            num_tokens=num_tokens,
            token_dim=context_dim,
            dim=context_dim,
            depth=cfg.MODEL.DECODER.DEPTH,
            heads=cfg.MODEL.DECODER.HEADS,
            mlp_dim=cfg.MODEL.DECODER.DIM,
            context_dim=context_dim
        )

        self.Decoder = TransformerDecoder(**transformer_decoder_args)
        self.n_views = cfg.DATASET.NUM_VIEWS

        self.pos_embed = SinePositionalEncoding3D(context_dim=context_dim, normalize=True)
        self.pos_embed_2d = PositionEmbeddingSine(context_dim=context_dim, normalize=True)
        
        # body_pose and betas (Neural Optimization)
        self.avgpool_smpl = nn.AvgPool2d((16, 12), stride=1)

        self.decpose_smpl = nn.Linear(context_dim, npose)
        self.decshape_smpl = nn.Linear(context_dim, 10)
        ###################################################

        self.smpl = SMPL_orig(
            cfg.J_regressor,
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False,
        )

        self.fuse_encoder_train = not (len(self.cfg.MODEL.VIT.HM_PRETRAINED) > 0)
        print(f'Fuse Encoder Train : {self.fuse_encoder_train}', flush=True)
        print(f'Using Pseud Ground', flush=True)

    def forward(self, target, eval=False, epoch=0):
        with torch.no_grad():

            batch_size, n_view = target['inp'].shape[:2]

            # heatmap fusion
            out_dict = self.fuser(target)  # j2d for original image
            if self.cfg.MODEL.FUSE_TYPE == 'NO':
                j2d = out_dict['j2d_NoFuse']
                pgt = out_dict['j3d_NoFuse']
            elif self.cfg.MODEL.FUSE_TYPE == 'SCORE':
                j2d = out_dict['j2d_ScoreFuse']
                pgt = out_dict['j3d_ScoreFuse']

            j2d_pred = rearrange(j2d, 'b v c j -> b v j c')

            j2d_pred_crop = torch.einsum('bvij,bvkj->bvki', target['trans_ad'], j2d_pred) # <- [256 192]におさまるように変換
            
            heatmap_96x96, target['joints_vis'] = batch_heatmap_generator_vis(
                j2d_pred_crop, 
                target['joints_vis'], 
                self.cfg.DATASET.HEATMAP_SIZE, 
                self.cfg.DATASET.AD_IMG_SIZE, 
                self.cfg.DATASET.sigma
                )
                    
            heatmap_96x96 = rearrange(heatmap_96x96, "b v j h w -> (b v) j h w")
            inp_size = self.cfg.MODEL.VIT.HM_SIZE
            heatmap = F.resize(img=heatmap_96x96, size=(inp_size, inp_size))[..., 32:-32]

            # image feature
            vit_inp = rearrange(target['ad_inp'], 'b v c h w -> (b v) c h w')
            img_feat = self.img_backbone(vit_inp[..., 32:-32])    # (batch * n_view 1280 16 12)

        # HeatEncoder
        with torch.set_grad_enabled(self.fuse_encoder_train):
            hm_feat = self.fuse_encoder(heatmap)                  # (batch * n_view 1024 16 12)
        
        # Estimate global orient
        with torch.no_grad():
            delta_global_orient, delta_cam = self.GCEstimator(img_feat)

        # Start Neural Optimization
        ret = []
        if eval:
            hm_ret = []
            hm_ret.append(heatmap)
        for iter in range(self.cfg.MODEL.ITERS + 1):
            if iter>0:
                with torch.set_grad_enabled(self.fuse_encoder_train):
                    hm_smpl_feat = self.fuse_encoder(heatmap_smpl)

                if self.cfg.MODEL.WITH_IMG:
                    kv_x = torch.cat([img_feat, hm_feat], dim=-3) # (batch * n_view C h w)   
                    q_x = torch.cat([img_feat, hm_smpl_feat], dim=-3)
                else:
                    kv_x = hm_feat
                    q_x = hm_smpl_feat
                
                # Type 1 query : Avgpooled after concatenating image features and heatmap features
                if self.query_type == 1:
                    q_x = self.avgpool_smpl(q_x)                  

                    kv_x = rearrange(kv_x, '(b v) c h w -> b v c h w', v=self.n_views)
                    q_x = rearrange(q_x, '(b v) c h w -> b v c h w', v=self.n_views)

                    kv_x = kv_x + self.pos_embed(kv_x)
                    q_x = q_x + self.pos_embed(q_x)

                    # (batch * n_view 1024) => (batch n_view(token) 1024)
                    kv_x_batch = rearrange(kv_x, 'b v c h w -> b (v h w) c', v=self.n_views)
                    q_x_batch = rearrange(q_x, 'b v c h w -> b (v h w) c', v=self.n_views)

                    query_out = torch.zeros_like(q_x_batch).float().to(self.cfg.DEVICE)
                    for v in range(self.n_views):
                        query_out[:, v:v+1] = self.Decoder(q_x_batch[:, v, None], context=kv_x_batch)

                    xc = query_out

                # Type 2 query : input to transformer decoder view by view
                elif self.query_type == 2:
                    kv_x = rearrange(kv_x, '(b v) c h w -> b v c h w', v=self.n_views)

                    kv_x = kv_x + self.pos_embed(kv_x)  # (batch view c h w)
                    q_x = q_x + self.pos_embed_2d(q_x)  # (batch * view c h w)

                    kv_x_batch = rearrange(kv_x, 'b v c h w -> b (v h w) c')
                    q_x_batch = rearrange(q_x, '(b v) c h w -> b v (h w) c', v=self.n_views)

                    query_out = torch.zeros_like(q_x_batch).float().to(self.cfg.DEVICE)
                    for v in range(self.n_views):
                        query_out [:, v:v+1] = self.Decoder(q_x_batch[:, v], context=kv_x_batch).unsqueeze(1)
                    
                    query_out = rearrange(query_out, 'b v (h w) c -> (b v) c h w', h=16)
                    xc = rearrange(self.avgpool_smpl(query_out).squeeze(), '(b v) c -> b v c',  b=batch_size)
                
                # Type 3 query : input to transformer decoder after concatenating image features and heatmap features
                elif self.query_type == 3:
                    kv_x = rearrange(kv_x, '(b v) c h w -> b v c h w', v=self.n_views)
                    q_x = rearrange(q_x, '(b v) c h w -> b v c h w', v=self.n_views)

                    kv_x = kv_x + self.pos_embed(kv_x)  # (batch view c h w)
                    q_x = q_x + self.pos_embed(q_x)  # (batch view c h w)

                    kv_x_batch = rearrange(kv_x, 'b v c h w -> b (v h w) c')
                    q_x_batch = rearrange(q_x, 'b v c h w -> b (v h w) c')

                    query_out = self.Decoder(q_x_batch, context=kv_x_batch)
                    query_out = rearrange(query_out, 'b (v h w) c -> (b v) c h w', v=self.n_views, h=16)
                    xc = rearrange(self.avgpool_smpl(query_out).squeeze(), '(b v) c -> b v c',  b=batch_size)

                global_orient = rearrange(pose, '(b v) p -> b v p', b=batch_size)[:, :, :6]
                body_pose = rearrange(pose, '(b v) p -> b v p', b=batch_size)[:, :, 6:]
                betas_bv = rearrange(betas, '(b v) p -> b v p', b=batch_size)
                cam_bv = rearrange(cam, '(b v) p -> b v p', b=batch_size)

                body_pose = body_pose + self.decpose_smpl(xc)
                betas_bv = betas_bv + self.decshape_smpl(xc)

                preds = {
                    'pose' : rearrange(torch.cat([global_orient, body_pose], dim=-1), 'b v p->(b v) p'),
                    'betas' : rearrange(betas_bv, 'b v n->(b v) n'),
                    'cam' : rearrange(cam_bv, 'b v p -> (b v) p')
                }

                # rot6d => rotmat
                global_orient_mat = rot6d_to_rotmat_g(preds['pose'][:, :6]).reshape(-1, 1, 3, 3)
                body_pose_mat = rot6d_to_rotmat(preds['pose'][:, 6:]).reshape(-1, 23, 3, 3)
                pose_mat = torch.cat([global_orient_mat, body_pose_mat], dim=1)

                preds['pose_mat'] = pose_mat

                smpl_out = self.smpl(
                    global_orient=pose_mat[:, :1],
                    body_pose=pose_mat[:, 1:],
                    betas=preds['betas'].reshape(-1, 10),
                    pose2rot=False
                )

                for key in smpl_out:
                    preds[key] = smpl_out[key]

                # update
                pose = preds['pose']
                betas = preds['betas']
                cam = preds['cam']

                # using pseudo-gt for alignment and heatmap generation
                pgt_3d = rearrange(pgt, 'b v j c -> (b v) j c') # if inference change to optimization or triangulation
                pred_3d = preds['regress_joints']

                # align for heatmap generation with pseudo-ground-truth(triangulation)
                transl = pgt_3d[:, 0] - pred_3d[:, 0]
                regress_3d = pred_3d + transl.unsqueeze(1)

                preds['pgt'] = pgt

                ret.append(preds)
            
            if iter == 0:
                heatmap_cur_smpl, pose, betas, cam, smpl_init = self.smpl.make_init_heatmap(
                    K=target['camera_param']['K'],
                    joints_3d_cam=pgt,
                    heatmap_size=self.cfg.DATASET.HEATMAP_SIZE,
                    image_size=self.cfg.DATASET.AD_IMG_SIZE,
                    sigma=self.cfg.DATASET.sigma,
                    trans=target['trans_ad'],
                    delta_global_orient=delta_global_orient,
                    delta_cam=delta_cam,
                    eval=eval,
                    g_rot=True
                )
                if eval:
                    smpl_init['pgt'] = pgt
            else:
                heatmap_cur_smpl = self.smpl.make_heatmap(
                    K=target['camera_param']['K'],
                    regress_joints=regress_3d,
                    heatmap_size=self.cfg.DATASET.HEATMAP_SIZE,
                    image_size=self.cfg.DATASET.AD_IMG_SIZE,
                    sigma=self.cfg.DATASET.sigma,
                    trans=target['trans_ad'],
                )
            
            # Joints Heatmap calculated from current smpl estimation (batch * n_view n_joints heat_h heat_w)
            heatmap_cur_smpl = rearrange(heatmap_cur_smpl, 'b v j h w -> (b v) j h w')

            inp_size = self.cfg.MODEL.VIT.HM_SIZE
            heatmap_smpl = F.resize(img=heatmap_cur_smpl, size=(inp_size, inp_size))[..., 32:-32]
            
            if iter>0 and self.cfg.LOSS.HEATMAP:
                preds['loss_heatmap'] = self.cfg.LOSS.HEATMAP[iter-1] * self.criterion[7](heatmap_96x96, heatmap_cur_smpl)

            if eval:
                hm_ret.append(heatmap_smpl)

        if not eval:
            return ret
        else:
            return ret, hm_ret, smpl_init
    
    def calculate_loss(self, preds_all, target):
        batch_size, n_view = target['inp'].shape[:2]

        preds = preds_all[-1] # Calculate loss for only final output

        preds['body_pose'] = rot6d_to_rotmat(preds['pose'][:, 6:]).reshape(-1, 23, 3, 3)
        preds['global_orient'] = rot6d_to_rotmat_g(preds['pose'][:, :6]).reshape(-1, 1, 3, 3)

        focal_length = self.cfg.FOCAL_LENGTH * torch.ones(preds['cam'].shape[0], 2).float().to(self.cfg.DEVICE)
        pred_cam_t = torch.stack([preds['cam'][:, 1], preds['cam'][:, 2], 2 * focal_length[:, 0] / (self.cfg.DATASET.IMG_SIZE * preds['cam'][:, 0] + 1e-9)], dim=-1)

        pred_j2d = perspective_projection(
            points=preds['regress_joints'],
            translation=pred_cam_t,
            focal_length=focal_length / self.cfg.DATASET.IMG_SIZE
        )

        # calculate loss
        # 3D
        pred_j3d = preds['regress_joints']
        gt_j3d = target['c_3d'].reshape(batch_size * n_view, -1, 3)
        loss_reg = self.cfg.LOSS.KP_3D * self.criterion[0](pred_j3d, gt_j3d)

        # 2D
        pred_j2d = pred_j2d.reshape(batch_size, n_view, -1, 2) * target['joints_vis'][..., None]
        gt_j2d = (target['c_2d_trans'] / self.cfg.DATASET.IMG_SIZE - 0.5) * target['joints_vis'][..., None]
        pred_j2d = pred_j2d.reshape(batch_size * n_view, -1, 2)
        gt_j2d = gt_j2d.reshape(batch_size * n_view, -1, 2)
        loss_2d = self.cfg.LOSS.KP_2D * self.criterion[1](pred_j2d, gt_j2d)

        # global_orient
        preds_global_orient = matrix_to_axis_angle(preds['global_orient']).reshape(batch_size * n_view, 1, 3)
        gt_global_orient = target['pose_c'][:, :, :3].reshape(batch_size * n_view, 1, 3)
        loss_global_orient = self.cfg.LOSS.GLOBAL_ORIENT * self.criterion[2](preds_global_orient, gt_global_orient)

        # body_pose
        preds_body_pose = matrix_to_axis_angle(preds['body_pose']).reshape(batch_size * n_view, 23, 3)
        gt_body_pose = target['pose_c'][:, :, 3:].reshape(batch_size * n_view , 23, 3)
        loss_body_pose = self.cfg.LOSS.POSE * self.criterion[3](preds_body_pose, gt_body_pose)

        # betas
        preds_beta = preds['betas'].reshape(batch_size * n_view, 10)
        gt_beta = target['shape_c'].reshape(batch_size * n_view, 10)
        loss_beta = self.cfg.LOSS.BETA * self.criterion[4](preds_beta, gt_beta)

        loss_dict = dict(
            loss_reg=loss_reg,
            loss_2d=loss_2d,
            loss_global_orient=loss_global_orient,
            loss_body_pose=loss_body_pose,
            loss_beta=loss_beta
        )

        # verts
        if self.cfg.LOSS.VERTS > 0.0:
            loss_verts = 0.0
            gt_verts = target['verts'].reshape(batch_size * n_view, -1, 3)
            if self.cfg.LOSS.VERTS_ALL:
                for phase in range(len(preds_all)-1):
                    preds_verts = preds_all[phase]['verts'].reshape(batch_size * n_view, -1, 3)
                    loss_verts = loss_verts + self.cfg.LOSS.VERTS * self.criterion[5](preds_verts, gt_verts)
            # final phase
            preds_verts = preds_all[-1]['verts'].reshape(batch_size * n_view, -1, 3)
            loss_verts = loss_verts + self.cfg.LOSS.VERTS * self.criterion[5](preds_verts, gt_verts)
            loss_dict['loss_verts'] = loss_verts
        
        # Normal Vector
        if self.cfg.LOSS.NORMAL_VECTOR > 0.0:
            loss_normal_vector = 0.0
            gt_verts = target['verts'].reshape(batch_size * n_view, -1, 3)
            if self.cfg.LOSS.NORMAL_VECTOR_ALL:
                for phase in range(len(preds_all)-1):
                    preds_verts = preds_all[phase]['verts'].reshape(batch_size * n_view, -1, 3)
                    loss_normal_vector = loss_normal_vector + self.cfg.LOSS.NORMAL_VECTOR * self.criterion[6](preds_verts, gt_verts)
            # final phase
            preds_verts = preds_all[-1]['verts'].reshape(batch_size * n_view, -1, 3)
            loss_normal_vector = loss_normal_vector + self.cfg.LOSS.NORMAL_VECTOR * self.criterion[6](preds_verts, gt_verts)
            loss_dict['loss_nv'] = loss_normal_vector

        
        return loss_dict