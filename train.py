import os
os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import time
import torch
import joblib
import random
import pprint
import numpy as np
import os.path as osp
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from lib.core.loss import get_loss
from lib.core.config import ROOT_DIR, parse_args
from lib.core.trainer import Trainer
from lib.models.smpl import get_smpl_faces
from lib.models.GCEstimator import GCEstimator
from lib.models.backbone.vit import get_vit
from lib.models.FuseEncoder import FuseEncoder
from lib.models.discriminator import Discriminator
from lib.models.adafuse import get_multiview_pose_net
from lib.utils.utils import create_logger, get_optimizer
from lib.models.HeatFormer import HeatFormerEncoderDecoder
from lib.dataset.data_loaders import get_loaders_multiview
from lib.models.backbone.pose_resnet_j33 import get_pose_net


def main(cfg, args):
    if cfg.SEED_VALUE >= 0:
        print(f'Seed value for the experiment {cfg.SEED_VALUE}', flush=True)
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)
    
    if cfg.LOGDIR:
        logger = create_logger(cfg.LOGDIR, phase='train')

        logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
        logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

        logger.info(pprint.pformat(cfg))
    else:
        logger = None

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    if cfg.LOGDIR:
        writer = SummaryWriter(log_dir=cfg.LOGDIR)
        writer.add_text('config', pprint.pformat(cfg), 0)
    else:
        writer = None

    heatmapper = get_pose_net(cfg).to(cfg.DEVICE)
    if cfg.POSE_RESNET.PRETRAINED:
        heatmapper_pretrain_path = osp.join(ROOT_DIR, cfg.POSE_RESNET.PRETRAINED)
        print(f'Load backbone pretrained model from {heatmapper_pretrain_path}', flush=True)
        print(flush=True)
        heatmapper_pretrained = torch.load(heatmapper_pretrain_path)
        heatmapper.load_state_dict(heatmapper_pretrained, strict=True)
    
    adafuse_model = get_multiview_pose_net(heatmapper, cfg).to(cfg.DEVICE)
    if cfg.MODEL.ADAFUSE.PRETRAINED:
        adafuse_pretrain_path = osp.join(ROOT_DIR, cfg.MODEL.ADAFUSE.PRETRAINED)
        print(f'Load adafuse pretrained model from {adafuse_pretrain_path}')
        print(flush=True)
        adafuse_pretrained = torch.load(adafuse_pretrain_path)
        adafuse_model.load_state_dict(adafuse_pretrained, strict=False)

    ### Load Backbone for Image
    img_vit = get_vit(cfg).to(cfg.DEVICE)
    GCEstimater = GCEstimator().to(cfg.DEVICE)
    img_vit_path = osp.join(ROOT_DIR, 'lib/models/pretrain/img_vit.pth.tar')
    gcestimator_path = osp.join(ROOT_DIR, 'lib/models/pretrain/gcestimator.pth.tar')
    img_vit_pretrained = torch.load(img_vit_path)
    gcestimator_pretrained = torch.load(gcestimator_path)
    img_vit.load_state_dict(img_vit_pretrained, strict=True)
    GCEstimater.load_state_dict(gcestimator_pretrained, strict=True)
    
    fuse_encoder = FuseEncoder(cfg).to(cfg.DEVICE)
    if cfg.MODEL.VIT.HM_PRETRAINED:
        HeatEncoder_path = osp.join(ROOT_DIR, cfg.MODEL.VIT.HM_PRETRAINED)
        print(f'Load img vit pretrained model from {HeatEncoder_path}', flush=True)
        print(flush=True)
        Encoder_pretrained_hm = torch.load(HeatEncoder_path)
        fuse_encoder.load_state_dict(Encoder_pretrained_hm, strict=True)

    criterion = get_loss(cfg.TRAIN.BATCH_SIZE, get_smpl_faces())

    model = HeatFormerEncoderDecoder(cfg, fuser=adafuse_model, fuse_encoder=fuse_encoder, img_backbone=img_vit, GCestimator=GCEstimater, criterion=criterion, query_type=cfg.MODEL.QUERY_TYPE).to(cfg.DEVICE)

    print(flush=True)
    print(f"######=====>The number of parameter:{sum([param.numel() for param in model.parameters()])}<======#####", flush=True)
    print(flush=True)

    optimizer = get_optimizer(
        model=model,
        optim_type=cfg.TRAIN.OPTIM,
        lr=cfg.TRAIN.LR,
        weight_decay=cfg.TRAIN.WD,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.TRAIN.LR_STEP,
        gamma=cfg.TRAIN.LR_FACTOR
        )

    # ========= Add Discriminator, optimizer, scheduler ========= #
    if cfg.LOSS.ADVERSARIAL>0:
        discriminator = Discriminator().to(cfg.DEVICE)
        optimizer_disc = get_optimizer(
            model=discriminator,
            optim_type=cfg.TRAIN.OPTIM,
            lr=cfg.TRAIN.DISC_LR,
            weight_decay=cfg.TRAIN.DISC_WD
            )
        lr_scheduler_disc = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_disc, 
            milestones=cfg.TRAIN.LR_STEP,
            gamma=cfg.TRAIN.LR_FACTOR
            )
    else:
        discriminator = None
        optimizer_disc = None
        lr_scheduler_disc = None

    # Data Loaders
    if not cfg.DEBUG:
        # It takes few minutes to load dataset
        train_loader, valid_loader = get_loaders_multiview(cfg)
    else:
        print(f'Loading fast dataset')
        start = time.time()
        train_loader = joblib.load(osp.join(ROOT_DIR, f'lib/dataset/easy/h36m_train_humans.pt'))[:cfg.DEBUG_ITER]
        valid_loader = joblib.load(osp.join(ROOT_DIR, f'lib/dataset/easy/h36m_valid_humans.pt'))[:cfg.DEBUG_ITER]
        print(f'### === > Loading Time : {time.time()-start} < === ###')

    print('------- Training Start -------', flush=True)

    Trainer(
        cfg=cfg,
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        discriminator=discriminator,
        optimizer=optimizer,
        optimizer_disc=optimizer_disc,
        epoch=cfg.EPOCH,
        lr_scheduler=lr_scheduler,
        lr_scheduler_disc=lr_scheduler_disc,
        device=cfg.DEVICE,
        writer=writer,
        debug=cfg.DEBUG,
        logdir=osp.join(ROOT_DIR, cfg.LOGDIR)
    ).fit()

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    cfg, cfg_file, args = parse_args()

    if not cfg.DRY_RUN:
        if not os.path.exists(cfg.LOGDIR):
            os.makedirs(cfg.LOGDIR)
    else:
        cfg.LOGDIR = ''
    main(cfg, args)