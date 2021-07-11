import os, sys
from token import NAME
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import pickle 
import glob
import cv2

import matplotlib.pyplot as plt

from .dataset import *

########## Slightly modified version of slightly modified version of LLFF data loading code 
########## see https://github.com/yenchenlin/nerf-pytorch and https://github.com/Fyusion/LLFF for original

#### Helper functions

def _minify(basedir, factors=[], resolutions=[], have_mask=False):
    '''resize images'''
    need_resize_img = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            need_resize_img = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            need_resize_img = True
    if not need_resize_img:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if is_image(f)]
    imgdir_orig = imgdir        

    if have_mask:
        maskdir = os.path.join(basedir, 'images_mask')
        masks = [os.path.join(maskdir, f) for f in sorted(os.listdir(maskdir))]
        maskdir_orig = maskdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int) or isinstance(r, float): # factor
            name = 'images_{}'.format(r)
            mask_name = 'images_mask_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else: # resolution
            name = 'images_{}x{}'.format(r[1], r[0])
            mask_name = 'images_mask_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        maskdir = os.path.join(basedir, mask_name)
        if os.path.exists(imgdir):
            continue
            
        print('Resizing images to size:', r)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        # resize images
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)        
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')

        # resize masks
        if have_mask:
            os.makedirs(maskdir)
            check_output('cp {}/* {} -r'.format(maskdir_orig, maskdir), shell=True)
            mask_ext = masks[0].split('.')[-1]
            args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(mask_ext)])
            print(args)
            os.chdir(maskdir)
            check_output(args, shell=True)
            os.chdir(wd)   

            if mask_ext != 'png':
                check_output('rm {}/*.{}'.format(maskdir, mask_ext), shell=True)
                print('Removed duplicates')

        print('Done')

def min_line_dist(rays_o, rays_d):
    A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
    b_i = -A_i @ rays_o
    pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
    return pt_mindist

def create_spheric_poses(radius, look_at=None, mean_phi=-np.pi/5, mean_y=0, n_poses=120):
    """
    Create circular poses around y axis.
    Inputs:
        radius: the radius of the circle.
        look_at: the look at position (center of the circle).
        mean_phi: the phi angle of the cameras.
        mean_y: the elevation of the cameras (height above the center of the circle).
        n_poses: number of poses
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius, look_at):
        
        trans_y_t = lambda y, t : np.array([
            [1,0,0,0],
            [0,1,0,y],
            [0,0,1,t],
            [0,0,0,1],
        ])     

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0, np.sin(th),0],
            [0,1,0,0],
            [-np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_y_t(mean_y, radius)
        if look_at is not None:
            c2w[:3, 3] += look_at
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, mean_phi, radius, look_at=look_at)] 

    return np.stack(spheric_poses, 0)
   
def recenter_poses(poses, pts):
    '''normalize the poses based on their positions and viewing directions'''
    def poses_avg(poses, is360=False):
        hwf = poses[0, :3, -1:]
        center = min_line_dist(poses[:, :3, 3:4], poses[:, :3, 2:3])
        center = poses[:, :3, 3].mean(0)
        z = poses[:, :3, 2].sum(0)
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([viewmatrix(z, up, center, is360=is360), hwf], 1)
        return c2w

    hwf = poses[:,:3,4:] 
    c2w = poses_avg(poses, True)[:3,:4]
    
    # Multiply poses with c2w
    pose_dtype = poses.dtype
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w_homo = np.concatenate([c2w, bottom], -2)
    c2w_inv = np.linalg.inv(c2w_homo)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)
    poses = c2w_inv @ poses
    poses = np.concatenate([poses[:,:3,:4], hwf], 2).astype(pose_dtype)

    if pts is not None:
        pts = np.concatenate([pts, pts[:,0:1]*0+1], axis=-1)[:,:,np.newaxis]
        pts = (c2w_inv @ pts)[:, :3,0]    

    return poses, pts, c2w

### LLFF data loader

class LLFFDataset(NeRFDataset):
    def _load_data(self):
        '''Load images/poses/masks/pointcloud from files'''
        print("Loading LLFF data from %s"%self.datadir)
        poses_arr = np.load(os.path.join(self.datadir, 'poses_bounds.npy'))
        if os.path.exists(os.path.join(self.datadir, 'pts.pkl')):
            pts_arr = pickle.load(open(os.path.join(self.datadir, 'pts.pkl'), "rb"))
        else:
            pts_arr = None

        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
        bds = poses_arr[:, -2:].transpose([1,0])
        
        sh = poses[:2,4,0].astype(np.uint32)
        
        sfx = ''
        
        if self.factor is not None and self.factor != 1:
            sfx = '_{}'.format(self.factor)
            _minify(self.datadir, factors=[self.factor], have_mask=self.have_mask)
            resized = True
        elif self.width is not None:
            self.factor = sh[1] / float(self.width)
            self.height = sh[0] * self.width // sh[1]
            _minify(self.datadir, resolutions=[[self.height, self.width]], have_mask=self.have_mask)
            sfx = '_{}x{}'.format(self.width, self.height)
            resized = True
        else:
            self.factor = 1
            resized = False
        
        imgdir = os.path.join(self.datadir, 'images' + sfx)
        assert(os.path.exists(imgdir)) # Image dir does not exist!

        if os.path.exists(os.path.join(self.datadir, 'image_names.pkl')):
            imgfiles = pickle.load(open(os.path.join(self.datadir, 'image_names.pkl'), "rb"))
            if resized:
                imgfiles = [os.path.splitext(f)[0]+".png" for f in imgfiles]
            imgfiles = [os.path.join(imgdir, f) for f in imgfiles]
        else:
            imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if is_image(f)]

        assert(poses.shape[-1] == len(imgfiles)) # Image/Pose number does not match!
        
        if resized:
            sh = imageio.imread(imgfiles[0]).shape
            poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
            poses[2, 4, :] = poses[2, 4, :] * 1./self.factor

        def imread(f):
            if f.endswith('png'):
                return imageio.imread(f, ignoregamma=True)
            else:
                return imageio.imread(f)
        def imgf2maskf(img_file):
            mask_dir = os.path.dirname(img_file).replace("images", "images_mask")
            img_id = os.path.basename(img_file).split(".")[0]
            if os.path.exists(os.path.join(mask_dir, "%s-removebg-preview.png"%img_id)):
                return os.path.join(mask_dir, "%s-removebg-preview.png"%img_id)
            else:
                return os.path.join(mask_dir, "%s.png"%img_id)
        def maskread(f, img, threshold=200):
            assert(os.path.exists(f))
            mask = imageio.imread(f)
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            return mask[...,3]>threshold # Add threshold to the mask

        # find largest size for padding
        padding_sh = poses[:2,4,:].max(axis=-1).astype(np.int32)
        # print(imgfiles)
        imgs = [imread(f)[...,:3]/255. for f in imgfiles]
        for i in range(len(imgs)):
            assert(imgs[i].shape[0] == poses[0, 4, i] and imgs[i].shape[1] == poses[1, 4, i]) # Image sizes and pose parameters don't match!
        print("Image size check finished.")

        # pad image with -1 to distinguish with valid pixels
        if self.have_mask:
            imgs_masks = [padding(maskread(imgf2maskf(f), imgs[i]), padding_sh) for i, f in enumerate(imgfiles)]
            imgs_masks = np.stack(imgs_masks, -1)
        else:
            imgs_masks = None
        imgs = np.stack([padding(x, padding_sh, pad_value=-1) for x in imgs], -1)  
        
        print('Loaded image data', imgs.shape, poses[:,-1,0])
        return poses, pts_arr, bds, imgs, imgs_masks, imgfiles

    def __init__(self, args, recenter=True, bd_factor=.75, path_zflat=False):    
        NeRFDataset.__init__(self, args)
        self.datadir = args.datadir
        self.factor = args.factor
        self.args = args
        self.N_test_pose = args.N_test_pose
        self.have_mask = args.have_mask

        self.width = None if args.width == 0 else args.width

        poses, pts, bds, imgs, imgs_masks, imgfiles = self._load_data() 
        print('Loaded', self.datadir, bds.min(), bds.max())
        
        # structure of poses_bound:
        # 3x5 matrix
        # first 3x4 is the transform matrix, in [-y, x, z] (or [down, right, backwards])
        # tha last column is [height, width, focal]

        # Correct rotation matrix ordering and move variable dim to axis 0
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # change [d, r, b] into [r, u, b]
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
        if imgs_masks is not None:
            imgs_masks = np.moveaxis(imgs_masks, -1, 0)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)
        
        poses, pts, avg_pose = recenter_poses(poses, pts)

        sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
        poses[:,:3,3] *= sc
        bds *= sc   

        if self.args.use_bbox:
            assert(pts is not None) # Point cloud is required to generate bounding box!
            pts *= sc
            pts_mean = pts.mean(axis=0)
            pts_norm = np.linalg.norm(pts-pts_mean, axis=-1)
            pts_norm_sorted = np.sort(pts_norm)
            pts_norm_thres = pts_norm_sorted[-max(5, len(pts_norm_sorted)//500)] # filter out outliers
            pts_reduced = pts[pts_norm <= pts_norm_thres]

            bound_box = np.stack([pts_reduced.min(axis=0), pts_reduced.max(axis=0)], axis=-1) # 3, 2
            bound_box_center = (bound_box[:,0]+bound_box[:,1])[:,None] / 2
            bound_box = bound_box_center + (bound_box-bound_box_center) * 1.3

            look_at = (bound_box[:,0]+bound_box[:,1]) / 2
            radius = np.linalg.norm((poses[:, :3, 3]-look_at[None,:]), axis=-1).mean() * 1.1
        else:
            bound_box = None
            look_at = None
            radius = np.linalg.norm(poses[:, :3, 3], axis=-1).mean() * 1.1

        mean_phi = np.arcsin(-poses[:,1,2]).mean()
        test_poses = create_spheric_poses(radius, look_at, mean_phi = mean_phi, 
                                            n_poses=self.N_test_pose)
        test_poses = np.array(test_poses).astype(np.float32)

        hwf = poses[0,:3,-1:].copy()
        hwf[2] = np.median(poses[:,2,-1:])
        test_poses = np.concatenate([test_poses, np.broadcast_to(hwf, test_poses[:,:3,-1:].shape)], -1)
            
        poses = torch.from_numpy(poses)
        test_poses = torch.from_numpy(test_poses)
        images = torch.from_numpy(imgs).float()
        if imgs_masks is not None:
            imgs_masks = torch.from_numpy(imgs_masks)

        if self.args.verbose:
            print('Data:')
            print(poses.shape, images.shape, bds.shape)

        i_test = np.arange(poses.shape[0])[args.test_offset::args.test_intv] 
        if self.args.verbose:
            print('HOLDOUT view is', i_test)
            print('HOLDOUT image_names is', [imgfiles[i] for i in i_test])

        hwf = poses[0,:3,-1] # intrinsic

        print('Loaded llff', images.shape, test_poses.shape, hwf, args.datadir)
        if isinstance(i_test, int):
            i_test = [i_test]

        i_val = i_test
        i_train = [i for i in range(len(images)) if (i not in i_test and i not in i_val)]
        if self.args.train_limit>0:
            g = np.random.RandomState(2022)
            train_idx = np.arange(len(i_train))
            train_idx = g.permutation(train_idx)
            i_train = [i_train[i] for i in train_idx[:self.args.train_limit]]
            if self.args.verbose:
                print("Training image limiting. Selected image indices are:", i_train)

        print('DEFINING BOUNDS')
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.2
        if not self.args.use_bbox:
            far = min(far, 8*near)
        test_img_id_file = os.path.join(args.datadir, 'test_img_id.txt')
        if os.path.exists(test_img_id_file):
            test_img_id = np.array([int(x) for x in open(test_img_id_file).readline()[:-1].split(' ')])
            test_img_name = [imgfiles[x] for x in test_img_id]
            if self.args.verbose:
                print("Test pair imgs: ", test_img_name)
        else:
            test_img_id = None

        print('NEAR FAR', near, far)

        self.images = images
        self.images_masks = imgs_masks
        self.poses = poses
        self.test_poses = test_poses
        self.bds = bds
        self.near = near
        self.far = far
        self.i_test = i_test
        self.i_val = i_val
        self.i_train = i_train
        self.bbox = bound_box
        self.test_img_id = test_img_id
        self.pts = pts

    def update_test_poses(self, train_poses):
        if self.args.use_bbox:
            bound_box = self.bbox
            look_at = (bound_box[:,0]+bound_box[:,1]) / 2
            radius = np.linalg.norm((train_poses[:, :3, 3]-look_at[None,:]), axis=-1).mean() * 1.1
        else:
            bound_box = None
            look_at = None
            radius = np.linalg.norm(train_poses[:, :3, 3], axis=-1).mean() * 1.1
        
        mean_phi = np.arcsin(-train_poses[:,1,2]).mean()
        test_poses = create_spheric_poses(radius, look_at, mean_phi = mean_phi, 
                                            mean_y = None, n_poses=self.N_test_pose)
        test_poses = np.array(test_poses).astype(np.float32)
        if self.args.test_resolution == -1:
            hwf = train_poses[:,:3,-1:].max(dim=0)
        else:
            hwf = train_poses[0,:3,-1:].copy()*0+self.args.test_resolution
        hwf[2] = np.median(train_poses[:,2,-1:])
        test_poses = np.concatenate([test_poses, np.broadcast_to(hwf, test_poses[:,:3,-1:].shape)], -1)
        test_poses = torch.from_numpy(test_poses)
        self.test_poses = test_poses


