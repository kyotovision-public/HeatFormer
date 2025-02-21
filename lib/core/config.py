# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import os.path as osp
import argparse
from yacs.config import CfgNode as CN

# CONSTANTS
cur_dir = os.getcwd()
ROOT_DIR = osp.join(cur_dir, 'HeatFormer')
BASE_DATA_DIR = osp.join(cur_dir, 'HeatFormer/data/base_data')
DB_DIR = osp.join(cur_dir, 'HeatFormer/data/preprocessed_data')
IMG_DIR = '/d/workspace/ymatsuda/dataset'
# IMG_DIR = osp.join(cur_dir, 'HeatFormer/data/dataset_img')
MPII3D_DIR = osp.join(cur_dir, 'mpi_inf_3dhp') # TODO change dir name
RICH_PATH = osp.join(cur_dir, 'RICH')
JOINT_REGRESSOR_TRAIN_EXTRA = osp.join(BASE_DATA_DIR, 'J_regressor_extra.npy')
SMPL_MEAN_PARAMS = osp.join(BASE_DATA_DIR, 'smpl_mean_params.npz')
SMPL_MODEL_DIR = BASE_DATA_DIR

H36M_kinematics = [[0, 1], [1, 2], [2, 3],
                   [0, 4], [4, 5], [5, 6],
                   [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 14], [14, 15], [15, 16],
                   [8, 11], [11, 12], [12, 13]]

# TODO remove not used for arguments
# Configuration variables
cfg = CN()
cfg.EPOCH = 30
cfg.DEBUG = True
cfg.DEBUG_ITER = 5
cfg.DEBUG_DIR = ''
cfg.DEVICE = 'cuda'
cfg.SEED_VALUE = 1
cfg.NUM_WORKERS = 16
cfg.LOGDIR = ''
cfg.DRY_RUN = True
cfg.FOCAL_LENGTH = 5000
cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = True
cfg.CUDNN.DETERMINISTIC = False
cfg.CUDNN.ENABLED = True
cfg.J_regressor = 'J_regressor_h36m.npy'

cfg.DATASET = CN()
cfg.DATASET.NUM_JOINTS = 17
cfg.DATASET.NUM_VIEWS = 4
cfg.DATASET.USE_VIEW_H36M = [1, 2, 3, 4]
cfg.DATASET.USE_VIEW_MPII3D = [0, 2, 7, 8]
cfg.DATASET.SCALE_FACTOR = 0.3
cfg.DATASET.ROT_FACTOR = 40
cfg.DATASET.sigma = 3
cfg.DATASET.AD_IMG_SIZE = 0
cfg.DATASET.TRAIN_SAMPLING_H36M = 10
cfg.DATASET.TRAIN_SAMPLING_MPII3D = 10
cfg.DATASET.VALID_SAMPLING = 10

cfg.DATASET.IMG_SIZE = 384
cfg.DATASET.HEATMAP_SIZE = 96
cfg.DATASET.FLIP = True

cfg.LOSS = CN()
cfg.LOSS.CUMSUM = False
cfg.LOSS.KP_2D = 0.0
cfg.LOSS.KP_3D = 300.0
cfg.LOSS.GLOBAL_ORIENT = 60.0
cfg.LOSS.POSE = 60.0
cfg.LOSS.BETA = 0.06
cfg.LOSS.ADVERSARIAL = 0.0 # 0.06
cfg.LOSS.VERTS = 0.0
cfg.LOSS.NORMAL_VECTOR = 0.0
cfg.LOSS.VERTS_ALL = False
cfg.LOSS.NORMAL_VECTOR_ALL = False
cfg.LOSS.HEATMAP = []

cfg.TRAIN = CN()
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.DATASETS = ['h36m', 'mpii3d']
cfg.TRAIN.EVAL_DATASETS = 'h36m'
cfg.TRAIN.LR_STEP = [20, 30, 40]
cfg.TRAIN.LR_FACTOR = 0.2
cfg.TRAIN.OPTIM = 'Adam'
cfg.TRAIN.LR = 0.00001
cfg.TRAIN.WD = 0.0
cfg.TRAIN.DISC_LR = 0.00001
cfg.TRAIN.DISC_WD = 0.0

cfg.MODEL = CN()
cfg.MODEL.TYPE = 'transformer_decoder_token'
cfg.MODEL.SMPL_HEAD_DIM = 1024
cfg.MODEL.ITERS = 3
cfg.MODEL.ADAFUSE = CN()
cfg.MODEL.ADAFUSE.PRETRAINED = ''
cfg.MODEL.HEATMAP_BACKBONE = CN()
cfg.MODEL.HEATMAP_BACKBONE.PRETRAINED = ''
cfg.MODEL.HEATMAP_BACKBONE.SIZE = 224
cfg.MODEL.VIT = CN()
cfg.MODEL.VIT.IMG_PRETRAINED = ''
cfg.MODEL.VIT.HM_PRETRAINED = ''
cfg.MODEL.VIT.HM_SIZE = 256
cfg.MODEL.HEATMAP = 'epipolar'
cfg.MODEL.BACKBONE_FREEZE = True
cfg.MODEL.FUSE_TYPE = 'NO'
cfg.MODEL.QUERY_TYPE = 1
cfg.MODEL.ALL_EST = False
cfg.MODEL.DIFF_V = True
cfg.MODEL.DIFF_ONLY = False
cfg.MODEL.PRETRAINED = ''
cfg.MODEL.WITH_IMG = True

cfg.MODEL.ENCODER = CN()
cfg.MODEL.ENCODER.DIM = 2048
cfg.MODEL.ENCODER.TOKEN_DIM = 512
cfg.MODEL.ENCODER.DEPTH = 6
cfg.MODEL.ENCODER.HEADS = 8

cfg.MODEL.DECODER = CN()
cfg.MODEL.DECODER.DIM = 2048
cfg.MODEL.DECODER.TOKEN_DIM = 512
cfg.MODEL.DECODER.DEPTH = 6
cfg.MODEL.DECODER.HEADS = 8

cfg.POSE_RESNET = CN()
cfg.POSE_RESNET.PRETRAINED = ''
cfg.POSE_RESNET.FINAL_CONV_KERNEL = 1
cfg.POSE_RESNET.DECONV_WITH_BIAS = False
cfg.POSE_RESNET.NUM_DECONV_LAYERS = 3
cfg.POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
cfg.POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
cfg.POSE_RESNET.NUM_LAYERS = 152

cfg.CAM_FUSION = CN()
cfg.CAM_FUSION.CROSSVIEW_FUSION = True

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./configs/config.yaml', help='cfg file path')
    parser.add_argument('--gpu', type=str, default='1', help='gpu num')

    args = parser.parse_args()
    print(args, end='\n\n')
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['EGL_DEVICE_ID'] = str(args.gpu)

    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()    

    return cfg, cfg_file, args
