import numpy as np
import os.path as osp

import cv2
import torch
import joblib

from torchvision import transforms
from lib.core.config import RICH_PATH, DB_DIR
from lib.utils.transform import get_affine_transform_cam

SET = 'test'

class RICH:
    def __init__(self, img_size, heatmap_size, ad_img_size):
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.ad_img_size = ad_img_size

        # TODO make db included seq name + frame
        self.db = joblib.load(osp.join(DB_DIR, 'rich_valid_db.pt'))

        self.num_view = 4

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.no_img_path = set(joblib.load(osp.join(DB_DIR, 'rich_not.pt')))

        self.offset = 0
    
    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        return self.get_single_idx(idx)

    def get_single_idx(self, idx):
        flag = True
        while flag:
            data = self.db[idx + self.offset]

            SET = 'test'
            SEQ_NAME = data['seq_name']
            frame = data['frame']
            cam_list = data['camera_list']

            for v in range(4):
                img_path = osp.join(RICH_PATH, SET, SEQ_NAME, f'cam_{cam_list[v]:02d}', f'{frame}_{cam_list[v]:02d}.jpeg')
                if img_path in self.no_img_path:
                    self.offset = self.offset + 1
                    break
            else:
                flag = False
        
        idx = idx + self.offset

        data = self.db[idx]

        SEQ_NAME = data['seq_name']
        frame = data['frame']
        cam_list = data['camera_list']
        cam_para = data['camera_param']
        j2d = data['c_2d']
        j3d = data['c_3d']

        cam_para['t'] = cam_para['t'].squeeze()

        SCENE_NAME, SUB_ID, _ = SEQ_NAME.split('_')
        SET = 'test'

        center_all = np.zeros((self.num_view, 2))
        scale_all = np.zeros((self.num_view, 2))
        scale_vit_all = np.zeros((self.num_view, 2))

        img_name = []

        # for adafuse
        crop_trans = np.zeros((self.num_view, 2, 3))
        aug_transes = np.zeros((self.num_view, 3, 3))
        inp_imgs_orig = np.zeros((self.num_view, self.img_size, self.img_size, 3))
        inp_ = np.zeros((self.num_view, 3, self.img_size, self.img_size))
        # target_heatmap = torch.zeros(self.num_view, self.num_joints, self.heatmap_size, self.heatmap_size).float()
        aff_t, inv_aff_t = [], []
        rotations = np.zeros(self.num_view)
        if self.ad_img_size>0:
            ad_inp_ = np.zeros((self.num_view, 3, self.ad_img_size, self.ad_img_size))
            crop_trans_ad = np.zeros((self.num_view, 2, 3))
        for i in range(self.num_view):
            img_path = osp.join(RICH_PATH, SET, SEQ_NAME, f'cam_{cam_list[i]:02d}', f'{frame}_{cam_list[i]:02d}.jpeg') 
            img_name.append(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # assert img.shape[0] == img.shape[1]

            j2d_v = j2d[i]

            bbox = [torch.min(j2d_v[:, 0]), torch.min(j2d_v[:, 1]), torch.max(j2d_v[:, 0]), torch.max(j2d_v[:, 1])]
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            scale_max = max((bbox[2] - bbox[0]) * 1.5 / 200, (bbox[3] - bbox[1]) * 1.5 / 200)
            scale_ori = np.array([scale_max, scale_max])
            scale_vit = self.expand_to_aspect_ratio(scale_ori) / 200

            trans = self.get_affine_transform(center, scale_ori, 0, (self.img_size, self.img_size))

            # full img -> cropped img
            input_img = cv2.warpAffine(img, trans, (self.img_size, self.img_size), flags=cv2.INTER_LINEAR)

            # transform
            inp = self.transform(input_img)

            if self.ad_img_size>0:
                trans_ad = self.get_affine_transform(center, scale_vit, 0, (self.ad_img_size, self.ad_img_size))
                ad_img = cv2.warpAffine(img, trans_ad, (self.ad_img_size, self.ad_img_size), flags=cv2.INTER_LINEAR)
                ad_inp = self.transform(ad_img)
            
            # full img -> heatmap
            affine_transfrom = get_affine_transform_cam(center, scale_ori, patch_size=(self.heatmap_size, self.heatmap_size), inv=False)
            inv_affine_transfrom = get_affine_transform_cam(center, scale_ori, patch_size=(self.heatmap_size, self.heatmap_size), inv=True)
            aff_trans = torch.eye(3).float()
            aff_trans[0:2] = torch.from_numpy(affine_transfrom).float()
            inv_aff_trans = torch.eye(3).float()
            inv_aff_trans = torch.from_numpy(inv_affine_transfrom).float()
            
            aug_trans = np.eye(3, 3)
            aug_trans[0:2] = trans # full img -> cropped img
            hm_scale = self.heatmap_size / self.img_size
            scale_trans = np.eye(3, 3)
            scale_trans[0, 0] = hm_scale
            scale_trans[1, 1] = hm_scale
            aug_trans = scale_trans @ aug_trans

            center_all[i] = center
            scale_all[i] = scale_ori
            scale_vit_all[i] = scale_vit

            crop_trans[i] = trans
            aug_transes[i] = aug_trans
            inp_imgs_orig[i] = input_img
            inp_[i] = inp
            rotations[i] = 0

            aff_t.append(aff_trans.unsqueeze(0))
            inv_aff_t.append(inv_aff_trans.unsqueeze(0))

            if self.ad_img_size>0:
                ad_inp_[i] = ad_inp
                crop_trans_ad[i] = trans_ad

        target = {
            'inp' : torch.from_numpy(inp_).float(),
            'trans' : torch.from_numpy(crop_trans).float(),
            'aug_trans' : torch.from_numpy(aug_transes).float(),
            'affine_trans' : torch.cat(aff_t, dim=0).float(),
            'inv_affine_trans' : torch.cat(inv_aff_t, dim=0).float(),
            'scale_ori' : torch.from_numpy(scale_all).float(),
            'scale_vit' : torch.from_numpy(scale_vit_all).float(),
            'center' : torch.from_numpy(center).float(),
            'rot' : torch.from_numpy(rotations).float(),
            'camera_param' : cam_para,                      # {'R':(nview 3 3), 't':(nview 3), 'K':(nview 3 3)}
            'c_3d' : j3d,                               # (nview 49 3) no aligned
            'c_2d' : j2d,
            'img_name': img_name,                     # (nview, ) list
            'joints_vis': torch.ones(4, 17)
        }

        if self.ad_img_size>0:
            target['ad_inp'] = torch.from_numpy(ad_inp_).float()
            target['trans_ad'] = torch.from_numpy(crop_trans_ad).float()

        return target

    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result
    
    def expand_to_aspect_ratio(self, cur_scale, target=[192, 256]):
        new_scale = np.zeros_like(cur_scale)
        cur_w, cur_h = cur_scale
        w = cur_w * 200
        h = cur_h * 200
        w_t, h_t = target

        if h * w_t < h_t * w:
            h_ = max(h_t * w / w_t, h)
            w_ = w
        else:
            h_ = h
            w_ = max(w_t * h / h_t, w)
        if h_ < h or w_ < w:
            import pdb;pdb.set_trace()
        
        new_scale = np.array([w_, h_])
        
        return new_scale

    def get_affine_transform(self,
                            center,
                            scale,
                            rot,
                            output_size,
                            shift=np.array([0, 0], dtype=np.float32),
                            inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale])

        scale_tmp = scale * 200.0
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans