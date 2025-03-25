import os
os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'

from lib.dataset.multiview_mpii3d import Multiview_MPII3D
from lib.dataset.multiview_h36m import Multiview_h36m
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
from tqdm import tqdm

import os.path as osp
import numpy as np
import torch
import random
import cv2
import os

align_list = ['gt', 'pgt', 'est']
ev_dataset_list = ['H36M', 'MPII3D']

cfg, cfg_file, args = parse_args_eval()

# For heatmap generation, you can use ground truth or pseudo-ground truth or estimation.
align_type = args.align_type  # select from align_list
assert align_type in align_list, 'Select from align_list: ("gt", "pgt", "est")'

# Output the result after each iteration
all_out = args.output_all

# Save flag and dir 
save = args.save
if save:
    save_freq = args.save_freq
    save_dir = osp.join(ROOT_DIR, args.save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

# Evaluation Dataset
ev_dataset = args.dataset
# we use 10 for Human3.6M and 5 for mpii3d
if ev_dataset == 'H36M':
    data_sampling = 10 
elif ev_dataset == 'MPII3D':
    data_sampling = 5

assert ev_dataset in ev_dataset_list, 'Select from ev_dataset_list: ("H36M", "MPII3D")'

ckpt_path = osp.join(ROOT_DIR, args.pretrain)

# Summary
print('='*15 + 'setting' + '='*15)
print(f'Use pretrain model: {ckpt_path}', flush=True)
print(f'align_type : {align_type}, dataset : {ev_dataset}', flush=True)
print(f'save : {save}, save_freq : {save_freq}, save_dir : {save_dir}')
print('='*37)

if align_type == 'pgt' or align_type == 'gt':
    assert cfg.MODEL.FUSE_TYPE == 'SCORE', 'Set cfg.MODEL.FUSE_TYPE "SCORE"'
else:
    assert cfg.MODEL.FUSE_TYPE == 'NO', 'Set cfg.MODEL.FUSE_TYPE "NO"'

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
    from lib.models.adafuse_for_eval import get_multiview_pose_net
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
elif align_type == 'est':
    from lib.models.HeatFormer_uncalib import HeatFormerEncoderDecoder

model = HeatFormerEncoderDecoder(cfg=cfg, fuser=adafuse_model, fuse_encoder=fuse_encoder, img_backbone=img_vit, GCestimator=GCEstimator, criterion=criterion, query_type=cfg.MODEL.QUERY_TYPE).to(cfg.DEVICE)

ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['model'], strict=True)
print(f'Loaded pretrained model from {ckpt_path}', flush=True)

if ev_dataset == 'MPII3D':
    valid_db = Multiview_MPII3D(
            load_opt='mpii3d_multiview', state='val', sampling_ratio=data_sampling, scale_factor=cfg.DATASET.SCALE_FACTOR, rot_factor=cfg.DATASET.ROT_FACTOR,
            sigma=cfg.DATASET.sigma, num_view=cfg.DATASET.NUM_VIEWS, cam_num=cfg.DATASET.NUM_VIEWS,
            use_view=cfg.DATASET.USE_VIEW_MPII3D, flip=True, img_size=cfg.DATASET.IMG_SIZE, heatmap_size=cfg.DATASET.HEATMAP_SIZE,
            additional_img_size=cfg.DATASET.AD_IMG_SIZE, swap=False
        )
elif ev_dataset == 'H36M':
    valid_db = Multiview_h36m(
            load_opt='h36m_multiview', state='val', sampling_ratio=data_sampling, scale_factor=cfg.DATASET.SCALE_FACTOR, rot_factor=cfg.DATASET.ROT_FACTOR,
            sigma=cfg.DATASET.sigma, num_view=cfg.DATASET.NUM_VIEWS, cam_num=cfg.DATASET.NUM_VIEWS,
            use_view=cfg.DATASET.USE_VIEW_H36M, flip=False, img_size=cfg.DATASET.IMG_SIZE,
            heatmap_size=cfg.DATASET.HEATMAP_SIZE, additional_img_size=cfg.DATASET.AD_IMG_SIZE, swap=False
        )
valid_loader = DataLoader(dataset=valid_db, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, drop_last=True, num_workers=cfg.NUM_WORKERS)
model.eval()

J_regressor_h36m = np.load(osp.join(BASE_DATA_DIR, 'J_regressor_h36m_correct.npy'))
smpl = SMPL_orig(cfg.J_regressor, SMPL_MODEL_DIR, batch_size=64, create_transl=False).to(cfg.DEVICE)

import time
timer_data = 0.0
timer_est = 0.0
start = time.time()
batch_generator = tqdm(valid_loader)

mpjpe_phase = [0.0] * cfg.MODEL.ITERS
pa_mpjpe_phase = [0.0] * cfg.MODEL.ITERS
mpvpe_phase = [0.0] * cfg.MODEL.ITERS
pck_phase = [0.0] * cfg.MODEL.ITERS
auc_phase = [0.0] * cfg.MODEL.ITERS
num = 0

def calc_metric(pred, gt, pred_v=None, gt_v=None):
    pred = pred - pred[:, 0, None]
    gt = gt - gt[:, 0, None]
    pred = torch.from_numpy(pred).float()
    ej = torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    sh = batch_compute_similarity_transform_torch(pred, gt)
    ep = torch.sqrt(((sh - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    if pred_v is not None:
        pred_v = pred_v - pred_v[:, 0, None]
        gt_v = gt_v - gt_v[:, 0, None]
        pred_v = torch.from_numpy(pred_v).float()
        ev = torch.sqrt(((pred_v - gt_v) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    
    if pred_v is not None:
        return ej, ep, ev
    else:
        return ej, ep, sh
    
def calc_metric_mpii3d(S1_hat, gt):
    err = torch.sqrt(((S1_hat - gt) ** 2).sum(dim=-1)) * 1000
    thresh = 150
    PCK = (err < thresh).sum()

    auc = 0
    for th in range(1, 151, 5):
        pck = (err < th).sum()
        auc = auc + pck

    return PCK, auc

for i,target in enumerate(batch_generator):
    move_dict_to_device(target, cfg.DEVICE)
    batch_size, n_view = target['inp'].shape[:2]
    timer_data = time.time() - start
    start = time.time()
    with torch.no_grad():
        preds_all = model(target)
    preds = preds_all[-1] # extract the results after final iteration

    timer_est = time.time() - start

    sample, n_kp, _ = preds['regress_joints'].shape
    pred_j3d = preds['regress_joints'].view(-1, n_kp, 3).detach().cpu().numpy()
    target_j3d = target['c_3d'].reshape(-1, n_kp, 3).detach().cpu().numpy()

    if ev_dataset == 'H36M':
        target_verts = target['verts'].view(-1, 6890, 3).cpu().numpy()
        pred_verts = preds['verts'].reshape(-1, 6890, 3).detach().cpu().numpy()

    if save and i%save_freq==0:
        idx = 0
        Intri = target['camera_param']['K'][idx].cpu().numpy()
        cam_mesh = preds['verts'].reshape(batch_size, n_view, -1, 3).detach().cpu().numpy()
        cam_joint = preds['regress_joints'].reshape(batch_size, n_view, -1, 3).detach().cpu().numpy()
        SAVE_PATH = osp.join(save_dir, f'result{i}.jpg')
        focal_length = cfg.FOCAL_LENGTH * torch.ones(preds['cam'].shape[0], 2).float().to(cfg.DEVICE)
        pred_cam_t = torch.stack([
            preds_all[-1]['cam'][:, 1], 
            preds_all[-1]['cam'][:, 2], 
            2 * focal_length[:, 0] / (cfg.DATASET.AD_IMG_SIZE * preds_all[-1]['cam'][:, 0] + 1e-9)
            ], dim=-1).reshape(batch_size, n_view, 1, 3).detach().cpu().numpy()

        save_images = []
        for number in range(cfg.DATASET.NUM_VIEWS):
            verts = cam_mesh[idx, number]

            if ev_dataset == 'H36M':
                print(osp.join('/d/workspace/ymatsuda/dataset', target['img_name'][number][idx][10:]))
                img = cv2.imread(osp.join('/d/workspace/ymatsuda/dataset', target['img_name'][number][idx][10:]))
            elif ev_dataset == 'MPII3D':
                print(osp.join('/d/workspace/ymatsuda/dataset/images_mpii3d', target['img_name'][number][idx][13:]))
                img = cv2.imread(osp.join('/d/workspace/ymatsuda/dataset/images_mpii3d', target['img_name'][number][idx][13:]))

            if align_type == 'est':
                img_orig = cv2.warpAffine(img, target['trans_ad'][idx, number].cpu().numpy(), (256, 256), flags=cv2.INTER_LINEAR)
                pred_cam_trans = pred_cam_t[idx, number]
                rendered_img = renderer(verts, pred_cam_trans[0].copy(), img_orig, scene_bg_color=(1,1,1))
                result_img = np.concatenate([img_orig, rendered_img], axis=0)
                save_images.append(result_img)
            else:
                j3d_pred = cam_joint[idx, number]

                if align_type == 'gt':
                    j3d_gt = target['c_3d'][idx, number].cpu().numpy()
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
        
        final_img = np.concatenate(save_images, axis=1)
        cv2.imwrite(SAVE_PATH, final_img)

    if ev_dataset == 'H36M':
        target_verts = target_verts - target_verts[:, 0, None]
        gt_v = torch.from_numpy(target_verts).float()

    target_j3d = target_j3d - target_j3d[:, 0, None]
    gt = torch.from_numpy(target_j3d).float()
    num = num + len(gt)

    for phase in range(cfg.MODEL.ITERS):
        pred_j3d_phase = preds_all[phase]['regress_joints'].view(-1, n_kp, 3).detach().cpu().numpy()
        if ev_dataset == 'H36M':
            pred_v = preds_all[phase]['verts'].view(-1, 6890, 3).detach().cpu().numpy()
            ej_phase, ep_phase, ev_phase = calc_metric(pred_j3d_phase, gt, pred_v, gt_v)
        elif ev_dataset == 'MPII3D':
            ej_phase, ep_phase, s1_hat = calc_metric(pred_j3d_phase, gt)
            PCK, AUC = calc_metric_mpii3d(s1_hat, gt)
        
        mpjpe_phase[phase] = mpjpe_phase[phase] + np.sum(ej_phase) * 1000
        pa_mpjpe_phase[phase] = pa_mpjpe_phase[phase] + np.sum(ep_phase) * 1000
        if ev_dataset == 'H36M':
            mpvpe_phase[phase] = mpvpe_phase[phase] + np.sum(ev_phase) * 1000
        elif ev_dataset == 'MPII3D':
            pck_phase[phase] = pck_phase[phase] + PCK
            auc_phase[phase] = auc_phase[phase] + AUC

    desc = f'({i+1}/{len(batch_generator)}) => '
    desc = desc + f'data: {timer_data:.5f} 'f'est: {timer_est:.5f} 'f'MPJPE: {mpjpe_phase[-1] / num:.5f} 'f'PA-MPJPE: {pa_mpjpe_phase[-1] / num:.5f} '
    if ev_dataset == 'H36M':
        desc = desc + f'MPVPE: {mpvpe_phase[-1] / num:.5f}'
    batch_generator.set_description(desc)
    start = time.time()

if all_out:
    for phase in range(cfg.MODEL.ITERS-1):
        if ev_dataset == 'H36M':
            print(f'PHASE {phase+1}  MPJPE : {mpjpe_phase[phase] / num:.4f}, PA-MPJPE : {pa_mpjpe_phase[phase] / num:.4f}, MPVPE : {mpvpe_phase[phase] / num:.4f}')
        elif ev_dataset == 'MPII3D':
            print(f'PHASE {phase+1}  PA-MPJPE : {pa_mpjpe_phase[phase] / num:.4f}, PCK : {pck_phase[phase] / num / n_kp:.4f}, AUC : {auc_phase[phase] / num / n_kp / 30:.4f}')

if ev_dataset == 'H36M':
    print(f'PHASE FINAL  MPJPE : {mpjpe_phase[-1] / num:.4f}, PA-MPJPE : {pa_mpjpe_phase[-1] / num:.4f}, MPVPE : {mpvpe_phase[-1] / num:.4f}')
elif ev_dataset == 'MPII3D':
    print(f'PHASE FINAl  PA-MPJPE : {pa_mpjpe_phase[-1] / num:.4f}, PCK : {pck_phase[-1] / num / n_kp:.4f}, AUC : {auc_phase[-1] / num / n_kp / 30:.4f}')