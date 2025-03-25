import os
os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from lib.dataset.multiview_BEHAVE import multiview_BEHAVE
from lib.models.backbone.vit import get_vit
from lib.models.backbone.pose_resnet_j33 import get_pose_net
from lib.core.config import parse_args_eval, ROOT_DIR, SMPL_MODEL_DIR, BASE_DATA_DIR
from lib.utils.utils import move_dict_to_device
from lib.models.smpl import SMPL_orig, get_smpl_faces
from lib.utils.render import render_mesh
from lib.utils.render_humans import Renderer
from lib.utils.eval_utils import batch_compute_similarity_transform_torch
from lib.models.FuseEncoder import FuseEncoder
from lib.models.GCEstimator import GCEstimator
from lib.core.loss import get_loss

from torch.utils.data import DataLoader
from einops import rearrange
from tqdm import tqdm

import os.path as osp
import numpy as np
import torch
import random
import cv2
import joblib
import os

align_list = ['gt', 'pgt']
score = 0.0
joints_vis = True

cfg, cfg_file, args = parse_args_eval()

# For heatmap generation, you can use ground truth or pseudo-ground truth or estimation.
align_type = args.align_type
assert align_type in align_list, 'Select from align_list: ("gt", "pgt")'

# Output the result after each iteration
all_out = args.output_all

# Save flag and dir
save = args.save
if save:
    save_freq = args.save_freq
    save_dir = osp.join(ROOT_DIR, args.save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

ckpt_path = osp.join(ROOT_DIR, args.pretrain)

# Summary
print('='*15 + 'setting' + '='*15)
print(f'Use pretrain model: {ckpt_path}', flush=True)
print(f'align_type : {align_type}, score : {score}', flush=True)
print(f'save : {save}, save_freq : {save_freq}, save_dir : {save_dir}')
print('='*37)

assert cfg.MODEL.FUSE_TYPE == 'SCORE', 'Set cfg.MODEL.FUSE_TYPE "SCORE"'

if cfg.SEED_VALUE >= 0:
    print(f'Seed value for the experiment {cfg.SEED_VALUE}', flush=True)
    os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
    random.seed(cfg.SEED_VALUE)
    torch.manual_seed(cfg.SEED_VALUE)
    np.random.seed(cfg.SEED_VALUE)

# Load Backbone
print(f'### === > Building Backbone < === ###')
print()

if align_type == 'pgt':
    from lib.models.adafuse_for_eval_mm import get_multiview_pose_net
else:
    from lib.models.adafuse import get_multiview_pose_net

heatmapper = get_pose_net(cfg).to(cfg.DEVICE)
adafuse_model = get_multiview_pose_net(heatmapper, cfg).to(cfg.DEVICE)
GCEstimator = GCEstimator().to(cfg.DEVICE)
img_vit = get_vit(cfg).to(cfg.DEVICE)
fuse_encoder = FuseEncoder(cfg).to(cfg.DEVICE)
criterion = get_loss(batch=cfg.TRAIN.BATCH_SIZE, face=get_smpl_faces())
renderer = Renderer(cfg, get_smpl_faces())

if align_type == 'pgt':
    from lib.models.HeatFormer_pseudo_gt import HeatFormerEncoderDecoder
elif align_type == 'gt':
    from lib.models.HeatFormer import HeatFormerEncoderDecoder

model = HeatFormerEncoderDecoder(cfg=cfg, fuser=adafuse_model, fuse_encoder=fuse_encoder, img_backbone=img_vit, GCestimator=GCEstimator, criterion=criterion, query_type=cfg.MODEL.QUERY_TYPE).to(cfg.DEVICE)

ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['model'], strict=True)
print(f'Loaded pretrained model from {ckpt_path}', flush=True)

view_index = [0,1,2,3]
valid_db = multiview_BEHAVE('valid', 384, 256, 96, score, view_index=[0,1,2,3])
valid_loader = DataLoader(dataset=valid_db, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, drop_last=True, num_workers=cfg.NUM_WORKERS)
model.eval()

J_regressor_openpose = torch.from_numpy(np.load(osp.join(BASE_DATA_DIR, 'J_regressor_body25.npy')))
J_regressor_openpose_batch = J_regressor_openpose[None].expand(cfg.TRAIN.BATCH_SIZE * 4, -1, -1)
J_regressor_openpose_batch_view = J_regressor_openpose[None, None].expand(cfg.TRAIN.BATCH_SIZE, 4, -1, -1)
smpl = SMPL_orig(cfg.J_regressor, SMPL_MODEL_DIR, batch_size=64, create_transl=False).to(cfg.DEVICE)

evaluation_accumulators = {'pred_j3d' : [], 'target_j3d' : [], 'jvs' : []}  


import time
timer_data = 0.0
timer_est = 0.0
start = time.time()
batch_generator = tqdm(valid_loader)

mpjpe_phase = [0.0] * cfg.MODEL.ITERS
pa_mpjpe_phase = [0.0] * cfg.MODEL.ITERS
num = 0

for i,target in enumerate(batch_generator):
    move_dict_to_device(target, cfg.DEVICE)
    batch_size, n_view = target['inp'].shape[:2]
    timer_data = time.time() - start
    start = time.time()
    with torch.no_grad():
        preds_all = model(target)
    preds = preds_all[-1] # extract the results after final iteration

    timer_est = time.time() - start
    pred_verts = preds['verts'].view(-1, 6890, 3).detach().cpu().numpy()
    pred_j3d = np.einsum('bij, bjk->bik', J_regressor_openpose_batch.cpu().numpy(), pred_verts)[:, :15]
    target_j3d = target['c_3d_gt'].reshape(-1, 25, 3).detach().cpu().numpy()[:, :15]

    if save and i%save_freq==0:
    # for idx in range(8):
        idx = 0
        Intri = target['camera_param']['K'][idx].cpu().numpy()
        cam_mesh = preds['verts'].reshape(batch_size, n_view, -1, 3).detach().cpu().numpy()
        if align_type == 'gt':
            cam_joint = np.einsum('bvij,bvjk->bvik', J_regressor_openpose_batch_view.cpu().numpy(), cam_mesh)
        elif align_type == 'pgt':
            cam_joint = preds['regress_joints'].reshape(batch_size, n_view, -1, 3).detach().cpu().numpy()
        SAVE_PATH = osp.join(save_dir, f'result{i}.jpg')
        
        save_images = []
        for number in range(cfg.DATASET.NUM_VIEWS):
            verts = cam_mesh[idx, number]
            j3d_pred = cam_joint[idx, number]

            img_path = osp.join(target['img_name'][idx], f'k{view_index[number]}.color.jpg')
            img = cv2.imread(img_path)

            if align_type == 'gt':
                j3d_gt = target['c_3d_gt'][idx, number].cpu().numpy()
                translation = j3d_gt[8] - j3d_pred[8]
            elif align_type == 'pgt':
                j3d_gt = preds['pgt'][idx, number].cpu().numpy()
                translation = j3d_gt[0] - j3d_pred[0]

            j3d_pred = j3d_pred + translation[None]
            verts = verts + translation[None]
                
            cam_para = {'focal': np.array([Intri[number,0,0], Intri[number,1,1]]), 'princpt': np.array([Intri[number,0,2], Intri[number,1,2]])}
            rendered_img = render_mesh(img, verts, smpl.faces, cam_para)
            img_crop = cv2.warpAffine(img, target['trans_ad'][idx, number].cpu().numpy(), (256, 256), flags=cv2.INTER_LINEAR)
            res_crop = cv2.warpAffine(rendered_img, target['trans_ad'][idx, number].cpu().numpy(), (256, 256), flags=cv2.INTER_LINEAR)
            result_img = np.concatenate([img_crop, res_crop], axis=0)
            save_images.append(result_img)

    if joints_vis:
        jv = rearrange(target['joints_vis_openpose'], 'b v j -> (b v) j')[:, :15, None].cpu()
    else:
        jv = None
    target_j3d = target_j3d - target_j3d[:, 8, None]

    gt = torch.from_numpy(target_j3d).float()
    num = num + len(gt)

    for phase in range(cfg.MODEL.ITERS):
        pred_verts_phase = preds_all[phase]['verts'].view(-1, 6890, 3).detach().cpu().numpy()
        pred_j3d_phase = np.einsum('bij, bjk->bik', J_regressor_openpose_batch.cpu().numpy(), pred_verts_phase)[:, :15]
        pred = pred_j3d_phase - pred_j3d_phase[:, 8, None]
        pred = torch.from_numpy(pred).float()
        S1_hat = batch_compute_similarity_transform_torch(pred, gt)

        if jv is not None:
            errors_j_b = torch.sqrt((((pred * jv) - (gt * jv)) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            errors_pa_b = torch.sqrt((((S1_hat * jv) - (gt * jv)) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        else:
            errors_j_b = torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
            errors_pa_b = torch.sqrt(((S1_hat - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

        mpjpe_phase[phase] = mpjpe_phase[phase] + np.sum(errors_j_b) * 1000
        pa_mpjpe_phase[phase] = pa_mpjpe_phase[phase] + np.sum(errors_pa_b) * 1000

    desc = f'({i+1}/{len(batch_generator)}) => '
    desc = desc + f'data: {timer_data:.5f} 'f'est: {timer_est:.5f} 'f'MPJPE: {mpjpe_phase[-1] / num:.5f} 'f'PA-MPJPE: {pa_mpjpe_phase[-1] / num:.5f} '
    batch_generator.set_description(desc)
    start = time.time()

print()
if all_out:
    for phase in range(cfg.MODEL.ITERS-1):
        print(f'PHASE {phase+1}  MPJPE : {mpjpe_phase[phase] / num:.4f}, PA-MPJPE : {pa_mpjpe_phase[phase] / num:.4f}')

print(f'PHASE FINAL  MPJPE : {mpjpe_phase[-1] / num:.4f}, PA-MPJPE : {pa_mpjpe_phase[-1] / num:.4f}')
