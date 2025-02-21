from lib.dataset.multiview_h36m import Multiview_h36m
from lib.dataset.multiview_mpii3d import Multiview_MPII3D

from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader


def get_loaders_multiview(cfg):
    batch_size = cfg.TRAIN.BATCH_SIZE
    train_db = []
    for dataset_name in cfg.TRAIN.DATASETS:
        if dataset_name == 'h36m':
            db = Multiview_h36m(
                load_opt='h36m_multiview', 
                state='train', 
                sampling_ratio=cfg.DATASET.TRAIN_SAMPLING_H36M, 
                scale_factor=cfg.DATASET.SCALE_FACTOR, 
                rot_factor=cfg.DATASET.ROT_FACTOR,
                sigma=cfg.DATASET.sigma,
                num_view=cfg.DATASET.NUM_VIEWS,
                cam_num=cfg.DATASET.NUM_VIEWS,
                use_view=cfg.DATASET.USE_VIEW_H36M,
                flip=cfg.DATASET.FLIP,
                img_size=cfg.DATASET.IMG_SIZE,
                heatmap_size=cfg.DATASET.HEATMAP_SIZE,
                additional_img_size=cfg.DATASET.AD_IMG_SIZE)
        elif dataset_name == 'mpii3d':
            db = Multiview_MPII3D(
                load_opt='mpii3d_multiview',
                state='train', 
                sampling_ratio=cfg.DATASET.TRAIN_SAMPLING_MPII3D, 
                scale_factor=cfg.DATASET.SCALE_FACTOR, 
                rot_factor=cfg.DATASET.ROT_FACTOR,
                sigma=cfg.DATASET.sigma,
                num_view=cfg.DATASET.NUM_VIEWS,
                cam_num=cfg.DATASET.NUM_VIEWS,
                use_view=cfg.DATASET.USE_VIEW_MPII3D,
                flip = not cfg.DATASET.FLIP,
                img_size=cfg.DATASET.IMG_SIZE,
                heatmap_size=cfg.DATASET.HEATMAP_SIZE,
                additional_img_size=cfg.DATASET.AD_IMG_SIZE
            )
        else:
            raise ValueError(f'{dataset_name} do not exists')
        train_db.append(db)

    train_db = ConcatDataset(train_db)

    train_loader = DataLoader(
        dataset=train_db,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=cfg.NUM_WORKERS,
    )

    if cfg.TRAIN.EVAL_DATASETS == 'h36m':
        valid_db = Multiview_h36m(
            load_opt='h36m_multiview', 
            state='val', 
            sampling_ratio=cfg.DATASET.VALID_SAMPLING, 
            scale_factor=cfg.DATASET.SCALE_FACTOR, 
            rot_factor=cfg.DATASET.ROT_FACTOR,
            sigma=cfg.DATASET.sigma,
            num_view=cfg.DATASET.NUM_VIEWS,
            cam_num=cfg.DATASET.NUM_VIEWS,
            use_view=cfg.DATASET.USE_VIEW_H36M,
            flip=cfg.DATASET.FLIP,
            img_size=cfg.DATASET.IMG_SIZE,
            heatmap_size=cfg.DATASET.HEATMAP_SIZE,
            additional_img_size=cfg.DATASET.AD_IMG_SIZE)
    elif cfg.TRAIN.EVAL_DATASETS == 'mpii3d':
        valid_db = Multiview_MPII3D(
            load_opt='mpii3d_multiview',
            state='val', 
            sampling_ratio=cfg.DATASET.VALID_SAMPLING, 
            scale_factor=cfg.DATASET.SCALE_FACTOR, 
            rot_factor=cfg.DATASET.ROT_FACTOR,
            sigma=cfg.DATASET.sigma,
            num_view=cfg.DATASET.NUM_VIEWS,
            cam_num=cfg.DATASET.NUM_VIEWS,
            use_view=cfg.DATASET.USE_VIEW_MPII3D,
            flip=not cfg.DATASET.FLIP,
            img_size=cfg.DATASET.IMG_SIZE,
            heatmap_size=cfg.DATASET.HEATMAP_SIZE,
            additional_img_size=cfg.DATASET.AD_IMG_SIZE
        )
    else:
        raise ValueError(f'{dataset_name} do not exists')
    
    valid_loader = DataLoader(
        dataset=valid_db,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=cfg.NUM_WORKERS,
    )

    return train_loader, valid_loader
