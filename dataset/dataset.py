import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

import utils.utils as utils

# Ray helpers

dataloader_rand = torch.Generator(2021)

def padding(img, sh, pad_value=0):
    if img.ndim == 2:
        return np.pad(img, [(0,sh[0]-img.shape[0]), (0, sh[1]-img.shape[1])], 'constant', constant_values=pad_value)
    else:
        return np.pad(img, [(0,sh[0]-img.shape[0]), (0, sh[1]-img.shape[1]), (0,0)], 'constant', constant_values=pad_value)

def is_image(f):
    f = f.lower()
    return f.endswith("jpg") or f.endswith("jpeg") or f.endswith("png")

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos, is360=False):
    if not is360: # fast-forward cameras, use mean backward(z) as the principle direction
        vec2 = normalize(z)
        vec1_avg = up
        vec0 = normalize(np.cross(vec1_avg, vec2))
        vec1 = normalize(np.cross(vec2, vec0))
    else: # 360 degree cameras, use mean up(y) as the principle direction
        vec1 = normalize(up)
        vec2_avg = z
        vec0 = normalize(np.cross(vec1, vec2_avg))
        vec2 = normalize(np.cross(vec0, vec1))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

'''
    Ray structure:
    o_x, o_y, o_z,
    dir_x, dir_y, dir_z,
    c_r, c_g, c_b,
    img_id, is_bg, -,
    n_x, n_y, n_z,
    depth_x, depth_y, depth_z (reserved)
'''
class NeRFDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.split = args.split
        self.N_ray = args.N_rand

        # testing params
        self.test_img_id = None
        self.env_sh = None
        pass

    def generate_rays(self):
        print("Generate rays...")

        sh = self.poses[:,:2,4].max(axis=0)[0].long()
        rays = torch.stack([utils.get_pixels(p, sh[0], sh[1]) for p in self.poses], 0) # [N, 2(ro+rd), H, W, 3]

        print('done, concats')
        rays_rgb = torch.cat([rays, self.images[:,None]], 1) # [N, 3(ro+rd+rgb), H, W, 3]

        # ray status: [img id, is background, reserved]
        ray_status = torch.zeros_like(self.images) # [N, H, W, 3]
        ray_status[:, :, :, 0] += torch.arange(len(self.images))[:, None, None] # img_id
        
        rays_rgb_st = torch.cat([rays_rgb, ray_status[:,None]], 1) # [N, 4(ro+rd+rgb+id), H, W, 3]
        rays_rgb_st = rays_rgb_st.permute(0,2,3,1,4) # [N, H, W, 4(ro+rd+rgb+id), 3]
        rays_rgb_st = rays_rgb_st[self.i_train] # train images only
        
        if self.images_masks is not None:
            mask_train = self.images_masks[self.i_train]
            N, H, W, d1, d2 = rays_rgb_st.shape
            rays_rgb_st = rays_rgb_st.reshape(N, H*W, d1, d2)
            mask_train = mask_train.reshape(N, H*W)
            self.rays_rgb_st_fg = torch.cat([rays_rgb_st[i,mask_train[i]] for i in range(N)], 0)
            self.rays_rgb_st_bg = torch.cat([rays_rgb_st[i,~mask_train[i]] for i in range(N)], 0)
            
            self.rays_rgb_st_fg = self.rays_rgb_st_fg[self.rays_rgb_st_fg[:,2,0]>=0]
            self.rays_rgb_st_bg = self.rays_rgb_st_bg[self.rays_rgb_st_bg[:,2,0]>=0]
            
            # force background to be white
            self.rays_rgb_st_bg[:, 2] = self.rays_rgb_st_bg[:, 2]*0+1
            # background=1 / foreground=0
            self.rays_rgb_st_bg[:, 3, 1] = 1
            self.rays_rgb_st_fg[:, 3, 1] = 0
            
            N_bg = min(self.rays_rgb_st_bg.shape[0], int(self.rays_rgb_st_fg.shape[0]*self.args.mask_ratio))
            bg_idx = torch.randperm(self.rays_rgb_st_bg.shape[0])[:N_bg]
            rays_rgb_st = torch.cat([self.rays_rgb_st_fg, self.rays_rgb_st_bg[bg_idx]], 0)
        else:
            #default setting: background mask is zero
            rays_rgb_st = rays_rgb_st.reshape([-1,4,3]) # [(N-1)*H*W, 4(ro+rd+rgb+id), 3]
            rays_rgb_st = rays_rgb_st[rays_rgb_st[:,2,0]>=0]

        self.cached_rays_rgb_st = rays_rgb_st.float()
        print('shuffle rays')
        self.shuffle()

        print('done')
        i_batch = 0
        
    def load_rays_from_file(self, ft):
        rays_dict = pickle.load(open(ft, "rb"))                  
        batch_rays = rays_dict['rays'][0].cpu()
        img_status = torch.cat([rays_dict['img_id'], rays_dict['rays_is_bg'], rays_dict['img_id']*0], dim=-1)[0].cpu()
        target_s = rays_dict['gt_color'][0].cpu()
        target_n = rays_dict['normal_mean'][0].cpu()
        if rays_dict['depth_mean'].ndim == 3:
            depth_status = torch.stack([rays_dict['depth_mean'][...,0], rays_dict['depth_var'][...,0], rays_dict['depth_mean'][...,0]*0], dim=-1)[0].cpu()
        else:
            depth_status = torch.stack([rays_dict['depth_mean'], rays_dict['depth_var'], rays_dict['depth_mean']*0], dim=-1)[0].cpu()
        batch = torch.cat([batch_rays, target_s[None,...], img_status[None,...], target_n[None,...], depth_status[None,...]], dim=0) # 2+1+1+2, B, 3
        self.cached_rays_rgb_st = batch.transpose(0, 1)
        if self.images_masks is not None:
            self.rays_rgb_st_bg = self.cached_rays_rgb_st[self.cached_rays_rgb_st[:, 3, 1]==1]
            self.rays_rgb_st_fg = self.cached_rays_rgb_st[self.cached_rays_rgb_st[:, 3, 1]==0]
        print('shuffle rays')
        self.shuffle()
    
    def shuffle(self):
        '''shuffle training rays and randomly pick background rays'''
        if self.images_masks is not None:        
            N_bg = min(self.rays_rgb_st_bg.shape[0], int(self.rays_rgb_st_fg.shape[0]*self.args.mask_ratio))
            bg_idx = torch.randperm(self.rays_rgb_st_bg.shape[0])[:N_bg]
            self.cached_rays_rgb_st = torch.cat([self.rays_rgb_st_fg, self.rays_rgb_st_bg[bg_idx]], 0)
        rand_idx = torch.randperm(self.cached_rays_rgb_st.shape[0])
        self.cached_rays_rgb_st = self.cached_rays_rgb_st[rand_idx]  

    def __len__(self):
        # this dataset has infinite length
        if self.split == 'val':
            return len(self.i_test)
        elif self.split == 'testtrain':
            return len(self.i_train)
        else:
            return max(1, len(self.cached_rays_rgb_st) // self.N_ray)

    def __getitem__(self, idx):
        if self.split == 'val':
            img_i = self.i_test[idx]
            target = self.images[img_i]
            target_mask = self.images_masks[img_i]
            sh = self.poses[img_i,:2,4].long()
            return {
                'poses': self.poses[img_i], # 3 x 5
                'gt_color': target[:sh[0],:sh[1]], # H x W x 3
                'gt_mask': target_mask[:sh[0],:sh[1]], # H x W
                'img_id': img_i if self.test_img_id is None else self.test_img_id[idx], # 1
            }
        elif self.split == 'testtrain':
            img_i = self.i_train[idx]
            target = self.images[img_i]
            target_mask = self.images_masks[img_i]
            sh = self.poses[img_i,:2,4].long()
            return {
                'poses': self.poses[img_i], # 3 x 5
                'gt_color': target[:sh[0],:sh[1]], # H x W x 3
                'gt_mask': target_mask[:sh[0],:sh[1]], # H x W
                'img_id': img_i, # 1
            }
        else:
            # Random over all images
            block_start = idx*self.N_ray % len(self.cached_rays_rgb_st)
            batch = self.cached_rays_rgb_st[block_start : block_start+self.N_ray] # [B, 2+1+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s, img_status = batch[:2], batch[2], batch[3]
            if batch.shape[0] > 4: # has geometry prior
                batch_normal, depth_status = batch[4], batch[5]
            else:
                batch_normal, depth_status = batch[2]*0, batch[3]*0

            return {
                'rays': batch_rays, # 2 x N_rays x 3
                'img_id': img_status[...,0:1], # N_rays x 1
                'rays_is_bg': img_status[...,1:2], # N_rays x 1
                'poses': self.poses[0], # 3 x 5
                'gt_color': target_s, # N_rays x 3
                'gt_normal': batch_normal, # N_rays x 3
                'depth_mean': depth_status[...,0:1], # N_rays x 1
                'depth_var': depth_status[...,1:2], # N_rays x 1
            }
    
    def print_info(self):
        print('Begin')
        print('TRAIN views are', self.i_train)
        print('TEST views are', self.i_test)
        print('VAL views are', self.i_val)
    
    def get_pose(self, idx):
        return self.poses[idx]

    def get_img(self, idx):
        return self.images[idx]

    def get_test_poses(self):
        return torch.FloatTensor(self.test_poses)

    def get_val_poses(self):
        # import pdb
        # pdb.set_trace()
        return torch.FloatTensor(self.poses[np.array(self.i_test)])

    def get_train_poses(self):
        # import pdb
        # pdb.set_trace()
        return torch.FloatTensor(self.poses[np.array(self.i_train)])

    def get_all_poses(self):
        # import pdb
        # pdb.set_trace()
        return torch.FloatTensor(self.poses)
        
    def update_test_poses(self, train_poses):
        pass