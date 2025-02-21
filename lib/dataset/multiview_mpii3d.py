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

from lib.core.config import DB_DIR, SMPL_MODEL_DIR, IMG_DIR, MPII3D_DIR
from lib.utils._kp_utils import convert_kps
from lib.utils.transform import get_affine_transform_cam

class Multiview_MPII3D:
    def __init__(
            self,
            load_opt : str, 
            state : str, 
            sampling_ratio : int,
            scale_factor : float = 0.0,
            rot_factor : float = 0.0,
            sigma : int = 3,
            num_view : int = 4, 
            cam_num : int = 4, # select 4 views from 8 views 
            use_view : list = [0, 2, 7, 8],
            dataset_name : str = 'mpii3d',
            swap : bool = True,
            flip : bool = False,
            img_size : int = 224,
            heatmap_size : int = 96,
            additional_img_size = 0,
            scale = 1.2,
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
        self.use_view = use_view
        self.num_joints = 17
        self.dataset_name = dataset_name
        self.scale = scale

        self.img_size = img_size
        self.ad_img_size = additional_img_size
        self.heatmap_size = heatmap_size
        self.counter = defaultdict(int)
        self.swap = swap
        self.debug = debug
        self.match = [[1,4], [2,5], [3,6]]
        self.flip = flip

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # MPII3D id
        self.train_id = [1,2,3,4,5,6,7]
        self.val_id = [8]

        # self.camera_id = [0, 2, 7, 8]

        print("### === > Loading Dataset < === ###", flush=True)
        print(flush=True)
        start = time.time()
        self.db, self.J3W, self.smpl_annot = self.load_db()
        print(f'### === > Loading Time : {time.time()-start} < === ###', flush=True)

        if self.dataset_name == 'mpii3d':
            start = time.time()
            self.mv_idx, self.order, self.cam_param = self.change_db_mpii()
            print(f'### === > Changing db Time : {time.time()-start} < === ###', flush=True)
        
        # filtering data
        start = time.time()
        del_set = []
        lack_cnt = 0
        self.cam_order = {}
        if self.dataset_name == 'mpii3d':
            for key in self.mv_idx:
                if len(self.mv_idx[key])<self.num_view:
                    del_set.append(key)
                    lack_cnt += 1
                    continue
                self.mv_idx[key].sort()
                self.cam_order[key] = [cam for cam, index in self.mv_idx[key]]
                # self.mv_idx[key] = [index for cam, index in self.mv_idx[key]]

        # delete
        for key in del_set:
            del self.mv_idx[key]
            self.order.remove(key)
        self.order = sorted(self.order)
        print(f'### === > Filtering Time : {time.time()-start} < === ###', flush=True)

        # Down Sampling
        if self.sampling_ratio>0:
            self.order = self.order[::sampling_ratio]
        
        # filtering missed smpl fitting
        if self.state == 'train':
            mis_fit = joblib.load(osp.join(DB_DIR, f'mis_fit_mpii3d_train_sampling_{sampling_ratio}.pt'))
            self.new_order = list()
            for key in self.order:
                if key in mis_fit:continue
                self.new_order.append(key)
            self.order = self.new_order

        print("lack count : ", lack_cnt, flush=True)
        print("state : ", "Train" if self.state=='train' else "Validate", flush=True)
        if self.dataset_name == 'mpii3d':
            print(f'{dataset_name} - number of dataset objects {self.__len__()}', flush = True)
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
        cam_select = [cam for cam, idx in self.mv_idx[self.order[idx]]]

        # video_3がない(ceiling camera?)ので-1する
        for i in range(self.num_view):
            cam_select[i] = int(cam_select[i])
            if cam_select[i]>3:
                cam_select[i] -= 1

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
        S, Seq, frame = self.order[idx]
        cam_params = self.cam_param[(S,Seq)]

        cam_para = {}
        cam_para['R'] = np.array([cam_params['R'][i] for i in cam_select])
        cam_para['t'] = np.array([cam_params['t'][i] for i in cam_select])
        cam_para['K'] = np.array([cam_params['K'][i] for i in cam_select])
        
        # 5. !!WORLD!! SMPL parameter gt (annotated by NeuralAnnot)
        pose_w = np.array(self.smpl_annot[S[-1]][Seq[-1]][str(int(frame))]['pose'])
        shape_w = np.array(self.smpl_annot[S[-1]][Seq[-1]][str(int(frame))]['shape'])

        # 6. No Aligned Joints 3D Camera Coords and World Coords
        pose, shape, j3d_cam = [], [], []
        for view in range(self.num_view):
            _, _, S_, Seq_, vid_, frame_ = img_path[view].split('.')[0].split('/')
            assert S == S_
            assert Seq == Seq_
            assert frame == frame_
            j3d_cam.append(torch.from_numpy(self.J3W[(S_[-1], Seq_[-1], vid_[-1], str(int(frame)-1))]).float().unsqueeze(0))
            root, _ = cv2.Rodrigues(pose_w[:3])
            root, _ = cv2.Rodrigues(np.dot(cam_para['R'][view], root))
            pose_cam = pose_w.copy()
            shape_cam = shape_w.copy()
            pose_cam[:3] = root.reshape(3)
            pose.append(torch.from_numpy(pose_cam).float().unsqueeze(0))
            shape.append(torch.from_numpy(shape_cam).float().unsqueeze(0))
        
        # (49, 3) -> (17, 3)  &  torch -> numpy
        c_3d = convert_kps(torch.cat(j3d_cam, dim=0).float(), 'spin', 'h36m')
        if self.flip:
            for view in range(self.num_view):
                for l,r in self.match:
                    c_3d[view, l], c_3d[view, r] = c_3d[view, r].copy(), c_3d[view, l].copy()

        pose = torch.cat(pose, dim=0).float()
        shape = torch.cat(shape, dim=0).float()

        res = self.smpl(
            global_orient=pose[:,:3].reshape(-1, 1, 3),
            body_pose=pose[:,3:].reshape(-1, 23, 3),
            betas=shape.reshape(-1,10)
        )

        verts = res.vertices.detach().cpu()  # (view 6890 3)

        c_3d_norm = c_3d / c_3d[..., -1:]
        c_2d = np.einsum('vij,vkj->vki', cam_para['K'], c_3d_norm)[..., :2]

        w_3d = np.einsum('vij,vkj->vki', np.linalg.inv(cam_para['R']), c_3d - cam_para['t'][:, None] / 1000).mean(axis=0)
    
        bbox = self.db['bbox'][db_indices]
        scale_ori = np.concatenate([(bbox[:, 2]/200)[:, None] * self.scale, (bbox[:, 3]/200)[:, None] * self.scale], axis=-1)
        center = np.concatenate([(bbox[:, 0])[:, None], (bbox[:, 1])[:, None]], axis=-1)
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
            path = osp.join(IMG_DIR, 'images_mpii3d', img_path[i][13:])
            img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            assert img.shape[0] == img.shape[1]

            if is_train:
                sf = self.scale_factor
                rf = self.rot_factor

                assert rotation == 0

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
                affine_c2d_ad = affine_c2d_ad / self.ad_img_size - 0.5
            
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
            'w_3d' : torch.from_numpy(w_3d[None]).float(),                          # (49 3) world coords
            'c_3d' : torch.from_numpy(c_3d).float(),                               # (nview 49 3) no aligned
            'c_2d' : torch.from_numpy(c_2d).float(),
            'c_2d_trans' : torch.from_numpy(c2d_transformed).float(),
            'img_name': list(img_path),                     # (nview, ) list
            'pose_c' : pose,
            'shape_c' : shape,
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
        w_3d  :  torch.Size([1, 17, 3])
        c_3d  :  torch.Size([4, 17, 3])
        c_2d  :  torch.Size([4, 17, 2])
        c_2d_trans  :  torch.Size([4, 17, 2])
        pose_c  :  torch.Size([4, 72])
        shape_c  :  torch.Size([4, 10])
        pose  :  torch.Size([72])
        shape  :  torch.Size([10])
        """

        return target

    def load_db(self):
        db_file = osp.join(DB_DIR, f'{self.dataset_name}_{self.state}_db.pt')
        if self.state == 'train':
            if self.load_opt == 'mpii3d_multiview':
                db_file = osp.join(DB_DIR, f'{self.dataset_name}_{self.state}_scale12_new_db.pt')
            else:
                raise NotImplementedError(f'{self.load_opt} is not implemented')
        elif self.state == 'val':
            if self.load_opt == 'mpii3d_multiview':
                db_file = osp.join(DB_DIR, f'{self.dataset_name}_{self.state}_scale12_new_db.pt')
            else:
                raise NotImplementedError(f'{self.load_opt} is not implemented')
        else:
            raise NotImplementedError(f'{self.state} is not appropriate. select train or val')

        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')
            
        self.db_file = db_file

        j3w = joblib.load(osp.join(DB_DIR, 'j3d_mpi_db.pt'))
        with open(osp.join(DB_DIR, 'extra_data/MPI-INF-3DHP_SMPL_NeuralAnnot.json')) as f:
            smpl_annot = json.load(f)

        print(f'Loaded {self.dataset_name} dataset from {db_file}', flush=True)
        return db, j3w, smpl_annot
    
    def change_db_mpii(self):
        """同一フレームのデータを取り出せるように整形"""
        seen = set()
        seen_set = set()
        order = set()
        cam_param = {}
        all_db = defaultdict(list)
        for idx,name in enumerate(self.db['img_name']):
            name = name.split('.')[0]
            _, _, S, Seq, vid, frame_idx = name.split('/')
            cam_idx = vid.split('_')[1]
            if int(cam_idx) not in self.use_view:continue
            all_db[(S, Seq, frame_idx)].append((cam_idx, idx))
            if (S, Seq, frame_idx) in seen:continue
            order.add((S,Seq,frame_idx))
            if (S, Seq) in seen_set:continue
            seen_set.add((S,Seq))
            cam_param[(S,Seq)] = self.get_cam_param_mpii(S, Seq)
        return all_db, order, cam_param
    
    def get_cam_param_mpii(self, S, Seq):
        calib_file = osp.join(MPII3D_DIR, f'{S}/{Seq}', 'camera.calibration')
        Ks, Rs, Ts = [], [], []
        file = open(calib_file, 'r')
        content = file.readlines()
        for vid_i in [0,1,2,4,5,6,7,8]:
            K = np.array([float(s) for s in content[vid_i * 7 + 5][11:-2].split()])
            K = np.reshape(K, (4, 4))[:3, :3]
            RT = np.array([float(s) for s in content[vid_i * 7 + 6][11:-2].split()])
            RT = np.reshape(RT, (4, 4))
            R = RT[:3, :3]
            T = RT[:3, 3]
            # T = RT[:3, 3] / 1000
            Ks.append(K)
            Rs.append(R)
            Ts.append(T)
        return {'K':np.array(Ks, dtype=np.float32),
                'R':np.array(Rs, dtype=np.float32), 
                't':np.array(Ts, dtype=np.float32)}
    
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
