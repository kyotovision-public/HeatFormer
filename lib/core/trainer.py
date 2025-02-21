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
import cv2
import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from torchvision import transforms

from progress.bar import Bar
from einops import rearrange
from tqdm import tqdm

from lib.utils.transform import Normalize
from lib.models.smpl import SMPL_orig
from lib.core.config import IMG_DIR, SMPL_MODEL_DIR
from lib.utils.utils import move_dict_to_device, AverageMeter
from lib.utils.render import render_mesh
from lib.utils.geometry import aa_to_rotmat
from lib.utils.eval_utils import batch_compute_similarity_transform_torch

logger = logging.getLogger(__name__)

class Trainer():
    def __init__(
            self,
            cfg,
            train_loader,
            valid_loader,
            model,
            discriminator,
            optimizer,
            optimizer_disc,
            epoch,
            lr_scheduler=None,
            lr_scheduler_disc=None,
            device=None,
            writer=None,
            debug=False,
            debug_freq=10,
            logdir='output',
            performance_type='min',
    ):
        
        # Configulation
        self.cfg = cfg

        # Data Loaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Models and optimizer
        self.model = model
        self.optimizer = optimizer

        # Discriminator and optimizer
        self.discriminator = discriminator
        self.optimizer_disc = optimizer_disc

        # Training parameters
        self.epoch = epoch
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_disc = lr_scheduler_disc
        self.device = device
        self.writer = writer
        self.debug = debug
        self.debug_freq = debug_freq
        self.logdir = logdir

        self.performance_type = performance_type
        self.train_global_step = 0
        self.best_performance = float('inf') if performance_type == 'min' else -float('inf')

        self._transform = transforms.Compose([
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        
        # Setting device if not
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.smpl = SMPL_orig(
            cfg.J_regressor,
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False,
        ).to(self.device)

        # Evaluation Metrics (MPJPE, PA-MPJPE, MPVPE)
        if self.cfg.TRAIN.EVAL_DATASETS == 'h36m':
            self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'pred_verts', 'target_verts'])
        else:
            self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'pred_verts'])

        # Setting Writer if not
        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            if self.logdir:
                self.writer = SummaryWriter(log_dir=self.logdir)

    def calculate_loss_disc(self, preds, target, batch_size, n_view):
        """Caluculate Loss for Discriminator"""
        gt_body_pose = target['pose_c'].reshape(batch_size * n_view, 24, 3)[:, 1:]
        gt_betas = target['shape_c'].reshape(batch_size * n_view, 10)
        pred_rotmat = preds['pose_mat'].reshape(batch_size * n_view, 24, 3, 3)[:, 1:]
        pred_betas = preds['betas'].reshape(batch_size * n_view, 10)

        BATCH = batch_size * n_view

        gt_rotmat = aa_to_rotmat(gt_body_pose.reshape(-1, 3)).reshape(batch_size * n_view, -1, 3, 3)
        disc_fake_out = self.discriminator(pred_rotmat.detach(), pred_betas.detach())
        loss_fake = ((disc_fake_out - 0.0)**2).sum() / BATCH
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0)**2).sum() / BATCH
        loss_disc = loss_fake + loss_real
        loss_d = self.cfg.LOSS.ADVERSARIAL * loss_disc

        disc_out = self.discriminator(pred_rotmat, pred_betas)
        loss_a = self.cfg.LOSS.ADVERSARIAL * ((disc_out - 1.0)**2).sum() / BATCH

        return loss_d, loss_a

    def train(self, epoch):
        losses = AverageMeter()

        timer = {
            'data': 0,
            'forward': 0,
            'loss': 0,
            'backward': 0,
            'batch': 0,
        }

        self.model.train()
        start = time.time()
        summary_string = ''
        bar = Bar(f'Epoch {epoch + 1}/{self.epoch}', fill='#', max=len(self.train_loader))

        print('\n', flush=True)
        print('='*10+'>'+'train'+'<'+'='*10, flush=True)
        batch_generator = tqdm(self.train_loader)
        for i,target in enumerate(batch_generator):
            move_dict_to_device(target, self.device)

            batch_size, n_view = target['inp'].shape[:2]

            timer['data'] = time.time() - start
            start = time.time()
            
            ### predict for all iterations
            preds_all = self.model(target, epoch=epoch)

            timer['forward'] = time.time() - start
            start = time.time()

            ### calculate loss
            loss_dic = self.model.calculate_loss(preds_all, target)

            timer['loss'] = time.time() - start
            start = time.time()

            loss_reg, loss_2d = loss_dic['loss_reg'], loss_dic['loss_2d']
            loss_go, loss_bp, loss_beta = loss_dic['loss_global_orient'], loss_dic['loss_body_pose'], loss_dic['loss_beta']
            
            loss = loss_reg + loss_2d + loss_go + loss_bp + loss_beta
        
            if self.cfg.LOSS.HEATMAP:
                loss_heat = 0.0
                loss_heat_db = []
                for preds_loss in preds_all:
                    loss_heat = loss_heat + preds_loss['loss_heatmap']
                    loss_heat_db.append(preds_loss['loss_heatmap'])
                loss = loss + loss_heat

            if self.cfg.LOSS.VERTS > 0.0:
                loss_verts = loss_dic['loss_verts']
                loss = loss + loss_verts
            
            if self.cfg.LOSS.NORMAL_VECTOR > 0.0:
                loss_normal_vector = loss_dic['loss_nv']
                loss = loss + loss_normal_vector

            if self.cfg.LOSS.ADVERSARIAL>0.0:
                loss_adv = 0.0
                loss_disc = 0.0
                for iters in range(len(preds_all)):
                    loss_d, loss_a = self.calculate_loss_disc(preds_all[iters], target, batch_size, n_view)
                    loss_disc = loss_disc + loss_d
                    loss_adv = loss_adv + loss_a
                loss = loss + loss_adv

            # <======= Backprop generator and discriminator
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.cfg.LOSS.ADVERSARIAL>0.0:
                self.optimizer_disc.zero_grad()
                loss_disc.backward()
                self.optimizer_disc.step()

            # <======= Log training info
            total_loss = loss

            losses.update(total_loss.item(), batch_size)

            timer['backward'] = time.time() - start
            timer['batch'] = timer['data'] + timer['forward'] + timer['loss'] + timer['backward']
            start = time.time()

            summary_string = f'({i + 1}/{len(self.train_loader)}) | Total: {bar.elapsed_td} | ' \
                             f'ETA: {bar.eta_td:}'
            
            loss_log = f'({i+1}/{len(batch_generator)}) => 'f'reg: {loss_reg:.4f} 'f'2d: {loss_2d:.4f} 'f'g-ori: {loss_go:.4f} 'f'pose: {loss_bp:.4f} 'f'beta: {loss_beta:.4f} '

            if self.cfg.LOSS.HEATMAP:
                for phase,h in enumerate(loss_heat_db):
                    loss_log = loss_log + f'h{phase}: {h:.4f} '

            if self.cfg.LOSS.VERTS > 0.0:
                loss_log = loss_log + f'verts: {loss_verts:.4f} '
            
            if self.cfg.LOSS.NORMAL_VECTOR > 0.0:
                loss_log = loss_log + f'normal: {loss_normal_vector:.4f} '
            
            if self.cfg.LOSS.ADVERSARIAL > 0.0:
                loss_log = loss_log + f'adv: {loss_adv:.4f} 'f'disc: {loss_disc:.4f}'
            
            batch_generator.set_description(loss_log)
            
            loss_dict = {
                'reg' : loss_reg,
                '2d' : loss_2d,
                'global_orient' : loss_go,
                'body_pose' : loss_bp,
                'beta' : loss_beta
                }
            
            if self.cfg.LOSS.HEATMAP:
                for phase,h in enumerate(loss_heat_db):
                    loss_dict[f'phase{phase}'] = h

            if self.cfg.LOSS.VERTS > 0.0:
                loss_dict['verts'] = loss_verts
            
            if self.cfg.LOSS.NORMAL_VECTOR > 0.0:
                loss_dict['normal'] = loss_normal_vector

            if self.cfg.LOSS.ADVERSARIAL > 0.0:
                loss_dict['adv'] = loss_adv
                loss_dict['disc'] = loss_disc
            
            if self.logdir:
                for k, v in loss_dict.items():
                    summary_string += f' | {k}: {v:.3f}'
                    self.writer.add_scalar('train_loss/'+k, v, global_step=self.train_global_step)

            for k,v in timer.items():
                summary_string += f' | {k}: {v:.2f}'

            if self.logdir:
                self.writer.add_scalar('train_loss/loss', total_loss.item(), global_step=self.train_global_step)

            self.train_global_step += 1
            bar.suffix = summary_string
            bar.next()

            if torch.isnan(total_loss):
                exit('Nan value in loss, exiting!...')
            # =======>

        bar.finish()

        logger.info(summary_string)

    def validate(self, epoch):
        self.model.eval()
        summary_string = ''
        bar = Bar('Validation', fill='#', max=len(self.valid_loader))

        if self.evaluation_accumulators is not None:
            for k,v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        print('\n', flush=True)
        print('='*10+'>'+'validate'+'<'+'='*10, flush='True')
        batch_generator = tqdm(self.valid_loader)
        for i, target in enumerate(batch_generator):
            move_dict_to_device(target, self.device)

            batch_size, n_view = target['inp'].shape[:2]
            with torch.no_grad():
                preds = self.model(target)[-1]

            sample, n_kp, _ = preds['regress_joints'].shape
            pred_j3d = preds['regress_joints'].view(-1, n_kp, 3).detach().cpu().numpy()
            target_j3d = target['c_3d'].reshape(-1, n_kp, 3).cpu().numpy()
            pred_verts = preds['verts'].view(-1, 6890, 3).detach().cpu().numpy()
            if self.cfg.TRAIN.EVAL_DATASETS == 'h36m':
                target_verts = target['verts'].view(-1, 6890, 3).cpu().numpy()

            self.evaluation_accumulators['pred_verts'].append(pred_verts)
            if self.cfg.TRAIN.EVAL_DATASETS == 'h36m':
                self.evaluation_accumulators['target_verts'].append(target_verts)
            self.evaluation_accumulators['pred_j3d'].append(pred_j3d)
            self.evaluation_accumulators['target_j3d'].append(target_j3d)


            summary_string = f'({i + 1}/{len(self.valid_loader)}) | ' \
                                f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'
            
            batch_generator.set_description(f'({i}/{len(batch_generator)}) => ')

            bar.suffix = summary_string
            bar.next()

        bar.finish()

        if self.logdir:
            logger.info(summary_string)

    def fit(self):
        """train and validate and evaluate"""
        for epoch in range(self.epoch):
            self.epoch = epoch
            self.train(epoch)
            self.validate(epoch)
            performance = self.evaluate()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)
            if self.lr_scheduler_disc is not None:
                self.lr_scheduler_disc.step(epoch)

            # log the learning rate
            if self.logdir:
                for param_group in self.optimizer.param_groups:
                    print(f'Learning rate {param_group["lr"]}')
                    self.writer.add_scalar('lr/gen_lr', param_group['lr'], global_step=self.epoch)

                logger.info(f'Epoch {epoch+1} performance: {performance:.4f}')

            self.save_model(performance, epoch)

        if self.logdir:
            self.writer.close()

    def save_model(self, performance, epoch):
        save_dict = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'performance': performance,
            'optimizer': self.optimizer.state_dict(),
            }
        
        if self.cfg.LOSS.ADVERSARIAL>0:
            save_dict['discriminator'] = self.discriminator.state_dict()
            save_dict['optimizer_disc'] = self.optimizer_disc.state_dict()

        filename = osp.join(self.logdir, 'checkpoint.pth.tar')
        torch.save(save_dict, filename)

        if self.performance_type == 'min':
            is_best = performance < self.best_performance
        else:
            is_best = performance > self.best_performance

        if is_best and self.logdir:
            logger.info('Best performance achived, saving it!')
            self.best_performance = performance
            shutil.copyfile(filename, osp.join(self.logdir, 'model_best.pth.tar'))

            with open(osp.join(self.logdir, 'best.txt'), 'w') as f:
                f.write(str(float(performance)))

    def resume_pretrained(self, model_path, epoch):
        if logger is not None:
            if osp.isfile(model_path):
                checkpoint = torch.load(model_path)
                self.start_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.best_performance = checkpoint['performance']

                logger.info(f"=> loaded checkpoint '{model_path}' "
                      f"(epoch {epoch}, performance {self.best_performance})")
            else:
                logger.info(f"=> no checkpoint found at '{model_path}'")
    
    def evaluate(self):
        print('\n', flush=True)
        print('='*10+'>'+'evaluating...'+'<'+'='*10, flush=True)
        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = np.vstack(v)

        pred_j3ds = self.evaluation_accumulators['pred_j3d']
        target_j3ds = self.evaluation_accumulators['target_j3d']

        pred_j3ds = torch.from_numpy(pred_j3ds).float()
        target_j3ds = torch.from_numpy(target_j3ds).float()

        pred_j3ds = pred_j3ds - pred_j3ds[:, 0, None]
        target_j3ds = target_j3ds - target_j3ds[:, 0, None]

        if self.cfg.TRAIN.EVAL_DATASETS == 'h36m':
            pred_mesh = self.evaluation_accumulators['pred_verts']
            target_mesh = self.evaluation_accumulators['target_verts']

            pred_mesh = torch.from_numpy(pred_mesh).float()
            target_mesh = torch.from_numpy(target_mesh).float()

            pred_mesh = pred_mesh - pred_mesh[:, 0, None]
            target_mesh = target_mesh - target_mesh[:, 0, None]
        
        print(f'Evaluating on {pred_j3ds.shape[0]} number of poses...', flush=True)

        batch, n_kp, _ = pred_j3ds.shape
        pred_j3ds = pred_j3ds.reshape(-1, n_kp, 3)
        target_j3ds = target_j3ds.reshape(-1, n_kp, 3)

        errors_j = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        if self.cfg.TRAIN.EVAL_DATASETS == 'h36m':
            errors_v = torch.sqrt(((pred_mesh - target_mesh) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
        errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

        m2mm = 1000

        mpjpe = np.mean(errors_j) * m2mm
        if self.cfg.TRAIN.EVAL_DATASETS == 'h36m':
            mpvpe = np.mean(errors_v) * m2mm
        pa_mpjpe = np.mean(errors_pa) * m2mm

        if self.cfg.TRAIN.EVAL_DATASETS == 'h36m':
            eval_dict = {
                'mpjpe': mpjpe,
                'pa-mpjpe': pa_mpjpe,
                'mpvpe': mpvpe
            }
        else:
            eval_dict = {
                'mpjpe': mpjpe,
                'pa-mpjpe': pa_mpjpe,
            }

        if self.cfg.DEBUG:
            string = ''
            for key in eval_dict:
                string += key.upper() + ' : ' + str(eval_dict[key]) + ' '
            print(string, flush=True)
                

        log_str = f'Epoch {self.epoch}, '
        log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])

        if self.logdir:
            logger.info(log_str)
            for k,v in eval_dict.items():
                self.writer.add_scalar(f'error/{k}', v, global_step=self.epoch)

        return pa_mpjpe
