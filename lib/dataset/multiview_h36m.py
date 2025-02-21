import os.path as osp
import numpy as np
import torch
import joblib
import json
import random
import cv2
import time

from smplx import SMPL
from collections import defaultdict
from torchvision import transforms

from lib.core.config import DB_DIR, SMPL_MODEL_DIR, IMG_DIR
from lib.utils._kp_utils import convert_kps
from lib.utils.transform import get_affine_transform_cam

class Multiview_h36m:
    def __init__(
            self, 
            load_opt : str, 
            state : str, 
            sampling_ratio : int, 
            scale_factor : float = 0.0,
            rot_factor : float = 0.0,
            sigma : int = 3,
            num_view : int = 4, 
            cam_num : int = 4, 
            use_view : list = [1, 2, 3, 4],
            dataset_name : str = 'h36m', 
            swap : bool = True, 
            flip : bool = True,
            img_size : int = 224,
            heatmap_size : int = 96,
            additional_img_size = 0,
            debug=False
        ) -> None:

        self.load_opt = load_opt
        self.state = state
        self.sampling_ratio = sampling_ratio
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.sigma = sigma
        self.cam_num = cam_num
        self.num_view = num_view
        self.use_view = set(use_view)
        self.num_joints = 17
        self.dataset_name = dataset_name
        self.debug = debug

        self.img_size = img_size
        self.ad_img_size = additional_img_size
        self.heatmap_size = heatmap_size
        self.counter = defaultdict(int)
        self.match = [[1,4], [2,5], [3,6]]
        self.swap = swap
        self.flip = flip

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Human3.6M id
        self.train_id = [1,5,6,7,8]
        self.val_id = [9,11]

        print("### === > Loading Dataset < === ###", flush=True)
        print(flush=True)
        start = time.time()
        self.db, self.J3W, self.smpl_w, self.ex_db = self.load_db(state)
        print(f'### === > Loading Time : {time.time()-start} < === ###', flush=True)

        if self.dataset_name == 'h36m':
            start = time.time()
            self.mv_idx, self.order, self.cam_param = self.change_db_h36m()
            print(f'### === > Changing db Time : {time.time()-start} < === ###', flush=True)
        
        # filtering data
        start = time.time()
        del_set = []
        lack_cnt = 0
        if self.dataset_name == 'h36m':
            for key in self.mv_idx:
                if len(self.mv_idx[key])!=self.num_view:
                    del_set.append(key)
                    lack_cnt += len(self.mv_idx[key])
                    continue
                sub, ac, sac, _ = key
                if self.isdamaged(int(sub), int(ac), int(sac)):
                    del_set.append(key)
                    lack_cnt += len(self.mv_idx[key])
                    continue
                self.mv_idx[key].sort()
        
        # delete
        for key in del_set:
            del self.mv_idx[key]
            self.order.remove(key)
        self.order = sorted(self.order)
        print(f'### === > Filtering Time : {time.time()-start} < === ###', flush=True)

        # Down Sampling
        if self.sampling_ratio>0:
            self.order = self.order[::sampling_ratio]

        print("lack count : ", lack_cnt, flush=True)
        print("state : ", "Train" if self.state=='train' else "Validate", flush=True)
        if self.dataset_name == 'h36m':
            print(f'{dataset_name} - number of dataset objects {self.__len__()}', flush=True)
        print(flush=True)
        
        self.smpl = SMPL(
            SMPL_MODEL_DIR,
            batch_size=64,
            create_transl=False
        )

    def __len__(self):
        return len(self.order)
    
    def __getitem__(self, idx):
        return self.get_single_idx(idx)
    
    def get_single_idx(self, idx):
        db_indices = [idx for cam, idx in self.mv_idx[self.order[idx]]]
        cam_select = [int(cam)-1 for cam, idx in self.mv_idx[self.order[idx]]]

        # random swap
        if self.swap:
            for i in range(len(db_indices)):
                for j in range(i+1, len(db_indices)):
                    if random.randint(1,100000)&1:
                        db_indices[i], db_indices[j] = db_indices[j], db_indices[i]
                        cam_select[i], cam_select[j] = cam_select[j], cam_select[i]

        assert len(db_indices) == self.num_view, print(f'db_indices : {db_indices}')
        assert len(cam_select) == self.num_view, print(f'cam_select : {cam_select}')

        self.counter[tuple(cam_select)] += 1

        is_train = (self.state == 'train')

        # 1. Load Img Path
        img_path = self.db['img_name'][db_indices]

        # 2. Load Key Points
        kp_2d = self.db['joints2D'][db_indices] # img corrds
        kp_3d = self.db['joints3D'][db_indices] # camera coords and pelvis aligned

        # 3. Load Img Feature
        feature = self.db['features'][db_indices]

        # 4. Load Camera Parameter and swap
        Subject, ac, subac, frame = self.order[idx]

        sub = str(int(Subject))
        ac = str(int(ac))
        subac = str(int(subac))
        frame = str(int(frame)-1)

        cam_params = self.cam_param[Subject]

        cam_para = {}
        cam_para['R'] = np.array([cam_params['R'][i] for i in cam_select])
        cam_para['t'] = np.array([cam_params['t'][i] for i in cam_select])
        cam_para['K'] = np.array([cam_params['K'][i] for i in cam_select])

        # 5. !!EACH CAMERA!! SMPL parameter gt (annotated by NeuralAnnot)
        pose = self.db['pose'][db_indices]
        shape = self.db['shape'][db_indices]

        res = self.smpl(
            global_orient=torch.from_numpy(pose[:,:3]).float().reshape(-1, 1, 3),
            body_pose=torch.from_numpy(pose[:,3:]).float().reshape(-1, 23, 3),
            betas=torch.from_numpy(shape).float().reshape(-1,10)
        )

        verts = res.vertices.detach().cpu()  # (view 6890 3)
        
        # 6. No Aligned Joints 3D Camera Coords and World Coords
        j3d_world = np.array(self.J3W[sub][ac][subac][frame], dtype=np.float32)/1000
        if self.flip:
            for l,r in self.match:
                j3d_world[l], j3d_world[r] = j3d_world[r].copy(), j3d_world[l].copy()
        j3d_world = torch.from_numpy(convert_kps(j3d_world[None], 'h36m', 'spin')).float()
        j3d_world_view = j3d_world.repeat(self.num_view,1,1)
        j3d_cam = torch.einsum('vij,vkj->vki', torch.Tensor(cam_para['R']).float(), j3d_world_view) + torch.Tensor(cam_para['t']).unsqueeze(1).float()/1000
        c_3d = convert_kps(j3d_cam.reshape(-1, 49, 3), 'spin', 'h36m').reshape(self.num_view, 17, 3)
        c_3d_norm = c_3d / c_3d[:, :, -1, None]
        c_2d = np.einsum('bij,bkj->bki', cam_para['K'], c_3d_norm)[:, :, :2]

        w_3d = convert_kps(j3d_world.mean(dim=0)[None], 'spin', 'h36m')

        pose_w = np.array(self.smpl_w[sub][ac][subac][frame]['pose'], dtype=np.float32)
        shape_w = np.array(self.smpl_w[sub][ac][subac][frame]['shape'], dtype=np.float32)

        bbox = self.ex_db['BBOX'][db_indices]
        scale_ori = self.ex_db['scale'][db_indices]
        center = self.ex_db['center'][db_indices]
        rotation = 0

        # Change scale for ViT
        scale_vit = self.expand_to_aspect_ratio(scale_ori) / 200

        if self.debug:
            target = {
                'camera_param' : cam_para,
                'c_3d' : torch.from_numpy(c_3d).float(),
                'verts' : verts,
                'img_name': list(img_path)
            }
            return target

        # for adafuse
        crop_trans = np.zeros((self.num_view, 2, 3))
        aug_transes = np.zeros((self.num_view, 3, 3))
        c2d_transformed = np.zeros((self.num_view, self.num_joints, 2))
        inp_imgs_orig = np.zeros((self.num_view, self.img_size, self.img_size, 3))
        inp_ = np.zeros((self.num_view, 3, self.img_size, self.img_size))
        joints_vis_ = []
        aff_t, inv_aff_t = [], []
        rotations = np.zeros(self.num_view)
        if self.ad_img_size>0:
            ad_inp_ = np.zeros((self.num_view, 3, self.ad_img_size, self.ad_img_size))
            ad_c2d_norm = np.zeros((self.num_view, self.num_joints, 2))
            crop_trans_ad = np.zeros((self.num_view, 2, 3))
            joints_vis_ad_ = []
        for i in range(len(img_path)):
            path = osp.join(IMG_DIR, img_path[i][10:])
            img : np.ndarray = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)[:1000]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            assert img.shape == (1000, 1000, 3)

            if is_train:
                sf = self.scale_factor
                rf = self.rot_factor

                assert rotation == 0

            # full img -> cropped img
            trans = self.get_affine_transform(center[i], scale_ori[i], rotation, (self.img_size, self.img_size))
            # trans = self.get_affine_transform(center[i], scale_vit[i], rotation, (self.img_size, self.img_size))

            # full img -> cropped img
            input_img = cv2.warpAffine(img, trans, (self.img_size, self.img_size), flags=cv2.INTER_LINEAR)

            # transform
            inp = self.transform(input_img)            

            # affine_c2d = affine_transform(c_2d[i], trans)
            affine_c2d = np.einsum('ij,kj->ki', trans, np.concatenate([c_2d[i], np.ones((c_2d[i].shape[0], 1))], axis=-1))
            joints_vis = np.ones(self.num_joints)
            out_of_range_idx = np.where((affine_c2d>=self.img_size) | (affine_c2d<0))
            if len(out_of_range_idx[0])>0:
                joints_vis[out_of_range_idx[:-1]] = 0

            if self.ad_img_size>0:
                trans_ad = self.get_affine_transform(center[i], scale_vit[i], rotation, (self.ad_img_size, self.ad_img_size))
                ad_img = cv2.warpAffine(img, trans_ad, (self.ad_img_size, self.ad_img_size), flags=cv2.INTER_LINEAR)
                ad_inp = self.transform(ad_img)
                affine_c2d_ad = np.einsum('ij,kj->ki', trans_ad, np.concatenate([c_2d[i], np.ones((c_2d[i].shape[0], 1))], axis=-1))
                joints_vis_ad = np.ones(self.num_joints)
                out_of_range_idx_ad = np.where((affine_c2d_ad>=self.ad_img_size) | (affine_c2d_ad<0))
                if len(out_of_range_idx_ad[0])>0:
                    joints_vis_ad[out_of_range_idx_ad[:-1]] = 0
                affine_c2d_ad = 2.0 * affine_c2d_ad / self.ad_img_size - 1.0
            
            # full img -> heatmap
            affine_transfrom = get_affine_transform_cam(center[i], scale_ori[i], patch_size=(self.heatmap_size, self.heatmap_size), inv=False)
            inv_affine_transfrom = get_affine_transform_cam(center[i], scale_ori[i], patch_size=(self.heatmap_size, self.heatmap_size), inv=True)
            # affine_transfrom = get_affine_transform_cam(center[i], scale_vit[i], patch_size=(self.heatmap_size, self.heatmap_size), inv=False)
            # inv_affine_transfrom = get_affine_transform_cam(center[i], scale_vit[i], patch_size=(self.heatmap_size, self.heatmap_size), inv=True)            
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

            crop_trans[i] = trans
            aug_transes[i] = aug_trans
            inp_imgs_orig[i] = input_img
            inp_[i] = inp
            rotations[i] = rotation
            c2d_transformed[i] = affine_c2d

            joints_vis_.append(torch.from_numpy(joints_vis).float().unsqueeze(0))
            aff_t.append(aff_trans.unsqueeze(0))
            inv_aff_t.append(inv_aff_trans.unsqueeze(0))

            if self.ad_img_size>0:
                ad_inp_[i] = ad_inp
                joints_vis_ad_.append(torch.from_numpy(joints_vis_ad).float().unsqueeze(0))
                ad_c2d_norm[i] = affine_c2d_ad
                crop_trans_ad[i] = trans_ad

        target = {
            'inp' : torch.from_numpy(inp_).float(),
            # 'img_trans_orig' : inp_imgs_orig, # original image after trans
            'trans' : torch.from_numpy(crop_trans).float(),
            'aug_trans' : torch.from_numpy(aug_transes).float(),
            'affine_trans' : torch.cat(aff_t, dim=0).float(),
            'inv_affine_trans' : torch.cat(inv_aff_t, dim=0).float(),
            'joints_vis' : torch.cat(joints_vis_, dim=0).float(),
            'scale_ori' : torch.from_numpy(scale_ori).float(),
            'scale_vit' : torch.from_numpy(scale_vit).float(),
            'center' : torch.from_numpy(center).float(),
            'rot' : torch.from_numpy(rotations).float(),
            'camera_param' : cam_para,                      # {'R':(nview 3 3), 't':(nview 3), 'K':(nview 3 3)}
            'w_3d' : torch.from_numpy(w_3d).float(),                                  # No Aligned World Coords (17 3)
            'c_3d' : torch.from_numpy(c_3d).float(),        # No Aligned Camera Coords (nvew 17 3)
            'c_2d' : torch.from_numpy(c_2d).float(),        # No Aligned Image Coords (nview 17 3)
            'c_2d_trans' : torch.from_numpy(c2d_transformed).float(),
            'img_name': list(img_path),                     # (nview, ) list
            'pose_c' : torch.from_numpy(pose).float(),
            'shape_c' : torch.from_numpy(shape).float(),
            'pose' : torch.from_numpy(pose_w).float(),
            'shape' : torch.from_numpy(shape_w).float(),
        }

        # if not is_train:
        if True:
            target['verts'] = verts

        if self.ad_img_size>0:
            target['ad_inp'] = torch.from_numpy(ad_inp_).float()
            target['c_2d_trans_ad'] = torch.from_numpy(ad_c2d_norm).float()
            target['joints_vis_ad'] = torch.cat(joints_vis_ad_, dim=0).float()
            target['trans_ad'] = torch.from_numpy(crop_trans_ad).float()
        """
        inp  :  torch.Size([4, 3, 384, 384])
        trans  :  torch.Size([4, 2, 3])
        aug_trans  :  torch.Size([4, 3, 3])
        affine_trans  :  torch.Size([4, 3, 3])
        inv_affine_trans  :  torch.Size([4, 2, 3])
        target_heatmap  :  torch.Size([4, 17, 96, 96])
        joints_vis  :  torch.Size([4, 17])
        scale  :  torch.Size([4, 2])
        center  :  torch.Size([4, 2])
        rot  :  torch.Size([4])
        camera_param R  :  (4, 3, 3)
        camera_param t  :  (4, 3)
        camera_param K  :  (4, 3, 3)
        features  :  torch.Size([4, 2048])
        w_3d  :  (1, 17, 3)
        c_3d  :  torch.Size([4, 17, 3])
        c_2d  :  torch.Size([4, 17, 2])
        c_2d_trans  :  torch.Size([4, 17, 2])
        pose_c  :  torch.Size([4, 72])
        shape_c  :  torch.Size([4, 10])
        pose  :  torch.Size([72])
        shape  :  torch.Size([10])
        """

        return target

    def load_db(self, state):
        if self.state == 'train':
            if self.load_opt == 'h36m_multiview':
                db_file = osp.join(DB_DIR, f'{self.dataset_name}_{self.state}_25fps_new_db.pt')
            else:
                raise NotImplementedError(f'{self.load_opt} is not implemented')
        elif self.state == 'val':
            if self.load_opt == 'h36m_multiview':
                db_file = osp.join(DB_DIR, f'{self.dataset_name}_test_25fps_new_db.pt')
            else:
                raise NotImplementedError(f'{self.load_opt} is not implemented')
        else:
            raise NotImplementedError(f'{self.state} is not appropriate. select train or val')

        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')
        
        # Init Pose and Shape
        if self.load_opt == 'h36m_multiview':
            joint_world = {}
            smpl_world = {}
            data_id = self.train_id if self.state == 'train' else self.val_id
            for id in data_id:
                with open(osp.join(DB_DIR, f'extra_data/Human36M_subject{id}_joint_3d.json')) as f:
                    joint_world[str(id)] = json.load(f)
                with open(osp.join(DB_DIR, f'extra_data/Human36M_subject{id}_SMPL_NeuralAnnot.json')) as f:
                    smpl_world[str(id)] = json.load(f)
            
        self.db_file = db_file

        print(f'Loaded {self.dataset_name} dataset from {db_file}', flush=True)

        if state=='train':
            ex_db = joblib.load(osp.join(DB_DIR, 'h36m_train_25fps_ex_db.pt'))
        else:
            ex_db = joblib.load(osp.join(DB_DIR, 'h36m_test_25fps_ex_db.pt'))

        return db, joint_world, smpl_world, ex_db
    
    def change_db_h36m(self):
        """同一フレームを取り出せるように変形する"""
        seen = set()
        seen_subject = set()
        order = set()
        cam_param = {}
        all_db = defaultdict(list)
        for idx, name in enumerate(self.db['img_name']):
            name : str = name.split('/')[-1].split('.')[0]
            _, subject_idx, _ , action_idx, _, subaction_idx, _, cam_idx, frame_idx = name.split('_')
            if int(cam_idx) not in self.use_view:continue
            all_db[(subject_idx, action_idx, subaction_idx, frame_idx)].append((cam_idx, idx))
            if (subject_idx, action_idx, subaction_idx, frame_idx) in seen:continue
            seen.add((subject_idx, action_idx, subaction_idx, frame_idx))
            order.add((subject_idx, action_idx, subaction_idx, frame_idx))
            if subject_idx in seen_subject:continue
            seen_subject.add(subject_idx)
            cam_param[subject_idx] = self.get_cam_param_h36m(subject_idx)
        
        return all_db, order, cam_param

    def get_cam_param_h36m(self,subject_idx):
        with open(osp.join(DB_DIR, f'Human36M_subject{int(subject_idx)}_camera.json'), 'rb') as f:
            cam = json.load(f)
        ret = {'R':[],'t':[],'K':[]}
        for i in range(4):
            data = cam[str(i+1)]
            R,t,f,c = np.array(data['R'],dtype=np.float32), np.array(data['t'],dtype=np.float32), np.array(data['f'],dtype=np.float32), np.array(data['c'],dtype=np.float32)
            K = np.zeros((3,3))
            K[0,0] = f[0]
            K[1,1] = f[1]
            K[2,2] = 1.
            K[0,2] = c[0]
            K[1,2] = c[1]
            ret['R'].append(R); ret['t'].append(t); ret['K'].append(K)
        ret['R'] = np.array(ret['R'])
        ret['t'] = np.array(ret['t'])
        ret['K'] = np.array(ret['K'])
        return ret
    
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
        for view in range(len(cur_scale)):
            cur_w, cur_h = cur_scale[view]
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
            
            new_scale[view] = np.array([w_, h_])
        
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
    
    def neighbor_cameras(self, d):
        """
        Args:
            d: dictionary of KRT
        Return:
            rank: dictionary of tuple of (list of cameras sorted by distance, distance)
        """
        cams = list(d.keys())
        centers = {}
        for k0, v0 in d.items():
            center, _ = self.camera_center(v0)
            centers[k0] = center
            assert len(center) == 3
        rank = {}
        # cntmeandist = 0
        for k0, v0 in centers.items():
            dist = {}
            for k1, v1 in centers.items():
                dist[k1] = np.linalg.norm(v0 - v1)
            r = sorted(cams, key=lambda x: dist[x])
            sorteddist = np.array(sorted(dist.values()))
            assert r[0] == k0
            # exclude the camera its own
            rank[k0] = (r[1:], sorteddist[1:])
        return rank
    
    def camera_center(self, KRT, engine='numpy'):
        """
        Args:
            KRT: 3x4
        """
        if engine == 'numpy':
            invA = np.linalg.inv(KRT[:, :3])
            return (-(np.dot(invA, KRT[:, 3]))), invA
        elif engine == 'torch':
            invA = torch.inverse(KRT[..., :3])
            center = -torch.matmul(invA, KRT[..., 3, None])
            out = torch.ones([center.shape[0], 4, 1], dtype=KRT.dtype, device=center.device)
            out[..., :3, :] = center
            return out, invA
        else:
            raise
    
    def isdamaged(self, subject, action, subaction):
        # from https://github.com/yihui-he/epipolar-transformers/blob/4da5cbca762aef6a89d37f889789f772b87d2688/data/datasets/joints_dataset.py#L174
        #damaged seq
        #'Greeting-2', 'SittingDown-2', 'Waiting-1'
        if subject == 11 and action == 2 and subaction == 2:
            return True
        if subject == 9:
            if action != 5 or subaction != 2:
                if action != 10 or subaction != 2:
                    if action != 13 or subaction != 1:
                        return False
        else:
            return False
        return True