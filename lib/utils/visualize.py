import os
import cv2
import torch
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from lib.core.config import ROOT_DIR

def visualize_heatmap(img_path : str, save_dir : str, save_name : str, joints_2d : torch.Tensor, heatmap : torch.Tensor, image_size : int, trans : torch.Tensor = None, all_kp=True):
    img = cv2.imread(img_path)

    joints_2d = joints_2d.numpy()
    heatmap = heatmap.numpy()
    trans = trans.numpy()

    img = cv2.warpAffine(img, trans, (image_size, image_size), flags=cv2.INTER_LINEAR)

    if not os.path.exists(osp.join(ROOT_DIR, save_dir)):
        os.mkdir(osp.join(ROOT_DIR, save_dir))

    if all_kp:
        for i in range(len(joints_2d)):
            x, y= joints_2d[i][:2]
            if x==0:continue
            cv2.circle(img, (int(x), int(y)), 3, (255,0,0), 3, cv2.LINE_8)
        if heatmap.shape[0] == 3:
            # 3 chanel
            plt.subplot(2, 2, 1)
            plt.imshow(img)
            plt.subplot(2, 2, 2)
            plt.imshow(heatmap[0])
            plt.subplot(2, 2, 3)
            plt.imshow(heatmap[1])
            plt.subplot(2, 2, 4)
            plt.imshow(heatmap[2])
        else:
            if heatmap.ndim == 3:
                heatmap = heatmap.max(axis=0)
            elif heatmap.ndim == 2:
                pass
            else:
                raise ValueError('heatmap dim is 2 or 3')   
            plt.subplot(1,2,1)
            plt.imshow(img)

            plt.subplot(1,2,2)
            plt.imshow(heatmap)

        print(osp.join(ROOT_DIR, save_dir, save_name))
        plt.savefig(osp.join(ROOT_DIR, save_dir, save_name))
    else:
        for i in range(len(joints_2d)):
            # plt.figure()
            ret = img.copy()
            x,y = joints_2d[i][:2]
            if x==0:continue
            cv2.circle(ret, (int(x), int(y)), 3, (255,0,0), 3, cv2.LINE_8)

            plt.subplot(1,2,1)
            plt.imshow(ret)

            plt.subplot(1,2,2)
            plt.imshow(heatmap[i])

            plt.savefig(osp.join(ROOT_DIR, save_dir, f'joints_{i}'))

