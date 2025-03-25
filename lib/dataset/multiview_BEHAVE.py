import joblib
import torch
import cv2

import os.path as osp
import numpy as np

from torchvision import transforms

from lib.core.config import DB_DIR, ROOT_DIR
from lib.utils.transform import get_affine_transform_cam


class multiview_BEHAVE:
    def __init__(self, state, img_size, ad_img_size, heatmap_size, score, num_view=4, view_index=[0,1,2,3]):

        self.state = state
        self.img_size = img_size
        self.ad_img_size = ad_img_size
        self.heatmap_size = heatmap_size
        self.score = score

        self.num_view = num_view
        self.view_index = view_index

        if state == 'train':
            self.db = joblib.load(osp.join(DB_DIR, 'BEHAVE_train_db.pt'))
        elif state == 'valid':
            self.db = joblib.load(osp.join(DB_DIR, 'BEHAVE_valid_db.pt'))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, idx):
        return self.get_single_idx(idx)

    def get_single_idx(self, idx):    
        data = self.db[idx]

        cam_para = {'R' : self.db[idx]['R'],
                    't' : self.db[idx]['t'],
                    'K' : self.db[idx]['K']}
        
        w_3d = self.db[idx]['w_3d_openpose']
        joints_vis = (w_3d[:, -1] >= self.score).int()
        w_3d_rep = w_3d[None].repeat(self.num_view, 1, 1)
        c_3d = torch.einsum('vij,vkj->vki', torch.from_numpy(cam_para['R']).float(), w_3d_rep[..., :3]) + torch.from_numpy(cam_para['t']).float().unsqueeze(1)
        j2d = torch.einsum('vij,vkj->vki', torch.from_numpy(cam_para['K']).float(), c_3d / c_3d[..., -1:])
        c_3d_dummy = c_3d.clone()
        c_3d_dummy[:, 0] = c_3d_dummy[:, 8]
        c_3d_dummy = c_3d_dummy[:, :17]

        center_all = np.zeros((self.num_view, 2))
        scale_all = np.zeros((self.num_view, 2))
        scale_vit_all = np.zeros((self.num_view, 2))

        # for adafuse
        crop_trans = np.zeros((self.num_view, 2, 3))
        aug_transes = np.zeros((self.num_view, 3, 3))
        inp_imgs_orig = np.zeros((self.num_view, self.img_size, self.img_size, 3))
        inp_ = np.zeros((self.num_view, 3, self.img_size, self.img_size))
        aff_t, inv_aff_t = [], []
        rotations = np.zeros(self.num_view)
        if self.ad_img_size>0:
            ad_inp_ = np.zeros((self.num_view, 3, self.ad_img_size, self.ad_img_size))
            crop_trans_ad = np.zeros((self.num_view, 2, 3))
        for i in range(self.num_view):
            img_path = osp.join(ROOT_DIR, data['img_name'], f'k{i}.color.jpg')
            # img_path = osp.join(data['img_name'], f'k{i}.color.jpg')
            img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # assert img.shape[0] == img.shape[1]

            j2d_v = j2d[i]
            valid_mask = j2d_v[:, 2] >= 0.5
            j2d_valid = j2d_v[valid_mask]

            if len(j2d_valid) == 0:
                return idx

            bbox = [torch.min(j2d_valid[:, 0]), torch.min(j2d_valid[:, 1]), torch.max(j2d_valid[:, 0]), torch.max(j2d_valid[:, 1])]
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
            scale_max = max((bbox[2] - bbox[0]) * 1.2 / 200, (bbox[3] - bbox[1]) * 1.2 / 200)
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
        
        cam_para_view = {}
        for key in cam_para.keys():
            cam_para_view[key] = cam_para[key][self.view_index]

        target = {
            'inp' : torch.from_numpy(inp_).float()[self.view_index],
            'trans' : torch.from_numpy(crop_trans).float()[self.view_index],
            'aug_trans' : torch.from_numpy(aug_transes).float()[self.view_index],
            'affine_trans' : torch.cat(aff_t, dim=0).float()[self.view_index],
            'inv_affine_trans' : torch.cat(inv_aff_t, dim=0).float()[self.view_index],
            'scale_ori' : torch.from_numpy(scale_all).float()[self.view_index],
            'scale_vit' : torch.from_numpy(scale_vit_all).float()[self.view_index],
            'center' : torch.from_numpy(center_all).float()[self.view_index],
            'rot' : torch.from_numpy(rotations).float()[self.view_index],
            'camera_param' : cam_para_view,                      # {'R':(nview 3 3), 't':(nview 3), 'K':(nview 3 3)}
            'w_3d' : w_3d[:, :4][self.view_index],                          # (49 3) world coords
            'c_3d' : c_3d_dummy[self.view_index],                               # (nview 49 3) no aligned
            'c_3d_gt' : c_3d[self.view_index],                               # (nview 49 3) no aligned
            'c_2d' : j2d[self.view_index],
            'img_name': self.db[idx]['img_name'],                     # (nview, ) list
            'joints_vis' : torch.ones(4, 17)[self.view_index],
            'joints_vis_openpose' : joints_vis[None].repeat(4,1)[self.view_index]
        }

        if self.ad_img_size>0:
            target['ad_inp'] = torch.from_numpy(ad_inp_).float()[self.view_index]
            target['trans_ad'] = torch.from_numpy(crop_trans_ad).float()[self.view_index]

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