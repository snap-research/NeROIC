
from abc import abstractmethod
from math import pi
import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.optim 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pytorch3d
from tqdm import tqdm, trange

import models.sh_functions as sh
import utils.utils as utils

# Helper functions
# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=cdf.device)

    u = u.contiguous()

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples

# BaseRenderer, used to preprocess and batchify rays
class BaseRenderer(nn.Module):
    def __init__(self, args):
        super(BaseRenderer, self).__init__()
        self.args = args
        self.netchunk = args.netchunk     
        self.model_type = args.model_type

        self.cam_dR = nn.Parameter(torch.zeros([args.N_vocab, 3]), requires_grad=True)
        self.cam_dt = nn.Parameter(torch.zeros([args.N_vocab, 3]), requires_grad=True)
        self.cam_df = nn.Parameter(torch.zeros([args.N_vocab, 2]), requires_grad=True)

    # function to render a batch of rays
    def _render_rays(self, ray_batch, testing, **kwargs):
        pass    

    def _render_rays_normal(self, ray_batch, **kwargs):
        pass

    def _render_sigma(self, pts):
        pass

    def calculate_loss(self):
        pass

    def init_cam_pose(self, poses):
        self.cam_R = poses[:, :3, :3].type_as(self.cam_dR) # rotation, n x 3 x 3
        self.cam_t = poses[:, :3, 3].type_as(self.cam_dt) # translation, n x 3
        self.cam_K = torch.from_numpy(utils.batched_hwf2mat(poses[:, :3, 4].cpu().numpy())).type_as(self.cam_df) # intrinsic parameters, n x 3

        self.cam_R = nn.Parameter(self.cam_R, requires_grad=False)
        self.cam_t = nn.Parameter(self.cam_t, requires_grad=False)
        self.cam_K = nn.Parameter(self.cam_K, requires_grad=False)

    def get_rotation(self, img_id):
        R = self.cam_R[img_id] # num_rays x 3 x 3
        dR = pytorch3d.transforms.axis_angle_to_matrix(self.cam_dR[img_id]) # num_rays x 3 x 3
        return torch.matmul(dR, R)

    def get_pose(self, img_id, hwf):
        R = self.get_rotation(img_id) # num_rays x 3 x 3
        t = self.cam_dt[img_id]+self.cam_t[img_id]
        hwf[2] += torch.mean(self.cam_df[img_id])
        return torch.cat([R, t[:, None], hwf[:, None]], dim=-1)

    def pix2rays(self, pixs, img_ids):
        ''' Turn pixels into 3d rays
        Args:
            pixs: [2, N_rays, 3]. Pixel coordinate of the pix, index:(W, H).
            img_ids: [N_rays]. index of image of each rays.
        Output:
            rays_o: [N_rays, 3]. origin of each ray.
            rays_d: [N_rays, 3]. direction of each ray.
        '''
        img_ids = img_ids.reshape(-1).long()
        i = pixs[0,:,0]
        j = pixs[0,:,1]
        K = self.cam_K[img_ids] # num_rays x 3 x 3
        R = self.cam_R[img_ids] # num_rays x 3 x 3
        t = self.cam_t[img_ids] # num_rays x 3
        df = self.cam_df[img_ids] # num_rays x 2
        dR = pytorch3d.transforms.axis_angle_to_matrix(self.cam_dR[img_ids]) # num_rays x 3 x 3
        dt = self.cam_dt[img_ids] # num_rays x 3

        dirs = torch.stack([(i-K[:,0,2])/(K[:,0,0]+df[:,0]), -(j-K[:,1,2])/(K[:,1,1]+df[:,1]), -torch.ones_like(i)], -1) # num_rays x 3
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.bmm(torch.bmm(dR, R), dirs[..., None])[...,0] # num_rays x 3
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = t+dt
        return torch.stack([rays_o, rays_d], dim=0)

    def get_depth_range(self, rays_o, rays_d, bbox):
        ''' Calculate the valid depth range of a ray given a bounding box.
        Args:
            rays_o: [N_rays, 3]. Origins of rays.
            rays_d: [N_rays, 3]. Directions of rays.
            bbox: Bounding box.
        Output:
            near: [N_rays].
            far: [N_rays].
        '''
        bbox = torch.from_numpy(bbox).type_as(rays_o).float()
        # disturb the ray a little bit to prevent zero division
        rays_d_nonzero = torch.where(rays_d.abs()<1e-9, rays_d*0+1e-9, rays_d) # N x 3
        p1 = (bbox[None,:,0] - rays_o) / rays_d_nonzero # N x 3
        p2 = (bbox[None,:,1] - rays_o) / rays_d_nonzero # N x 3
        near = torch.minimum(p1, p2).clamp(0, 1e2).max(dim=1)[0]
        far = torch.maximum(p1, p2).clamp(0, 1e2).min(dim=1)[0]
        return near.detach(), far.detach()

    def forward(self, pixel_coords=None, test_pose=None, img_id=0, chunk=1024*32, near=0., far=1., 
                    bbox=None, use_viewdirs=False, testing=False, normal_mode=False, light_param=None, **kwargs):
        """Render rays
        Args:
            pixel_coords: [2, batch_size, 3]. Pixel Coordinates for each example in batch.
            test_pose: [3, 5]. C2W Matrix of the test data.
            img_id: [batch_size]. Image index of each rays.
            chunk: int. Maximum number of rays to process simultaneously. Used to
                control maximum memory usage. Does not affect final results.
            near: [batch_size]. Nearest distance for a ray.
            far: [batch_size]. Farthest distance for a ray.
            bbox: [2, 3]. Bounding box of the object.
            use_viewdirs: bool. If True, use viewing direction of a point in space in model.
            testing: bool. If True, use testing mode.
            normal_mode: bool. If True, use normal_mode. (Priority: normal>testing>train)
            light_param: input lighting parameters. (See train.py)
        Returns:
            rgb_map: [batch_size, 3]. Predicted RGB values for rays.
            disp_map: [batch_size]. Disparity map. Inverse of depth.
            acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
            ...
        """

        if testing:
            assert(test_pose is not None) # test poses must be given
            rays_o, rays_d = utils.get_rays(test_pose.cpu()).to(test_pose.device)
            img_id = torch.zeros_like(rays_o[...,0:1])+img_id
        else:
            rays_o, rays_d = self.pix2rays(pixel_coords, img_id)

        if self.model_type== "rendering" or not self.args.optimize_camera:
            # Camera optimization turned off
            rays_o = rays_o.detach()
            rays_d = rays_d.detach()

        sh = rays_d.shape # [..., 3]
        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        img_id = torch.reshape(img_id, [-1,1]).float()

        if use_viewdirs:
            # Provide ray directions as input
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1,3]).float()

        # # ATTENTION: only support single camera intrinsic
        # if ndc:
        #     # for forward facing scenes
        #     rays_o, rays_d = utils.ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
        
        # Create ray batch

        # Calculate valid depth range of rays
        if bbox is not None:
            near_d, far_d = self.get_depth_range(rays_o, rays_d, bbox)
            near = torch.clamp(near_d, near)[:,None]
            far = torch.clamp(far_d, 0, far)[:,None]
        else:
            near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        rays = torch.cat([rays, img_id], -1)

        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        # Render and reshape
        # Render rays in smaller minibatches to avoid OOM.
        ret_dict = {}
        if normal_mode == True: # Extracting normal from autodiff
            ret_keys = ['acc_map', 'depth_mean', 'depth_var', 'normal_mean', 'pts_map']
        elif testing == True: # Only keep buffers that need to be printed out (to reduce memory consumption)
            ret_keys = ['rgb_map', 'acc_map', 'depth_map', 'static_rgb_map', 'normal_map_weighted', 'albedo_map', 'spec_map', 
                        'glossiness_map', 'rgb_map_coarse', 'transient_acc_map', 'static_only_acc_map', 'is_edge']
        else: # Return all possible buffers
            ret_keys = None

        rays_range = range(0, rays.shape[0], chunk)
        if (self.args.verbose and testing) or normal_mode: # print progress
            rays_range = tqdm(rays_range)
        for i in rays_range:
            # rendering 
            if normal_mode:
                ret = self._render_rays_normal(rays[i:i+chunk], **kwargs)
            else:                
                ret = self._render_rays(rays[i:i+chunk], testing=testing, light_param=light_param, **kwargs)

            for k in ret.keys():
                if ret_keys is not None and k not in ret_keys:
                    continue
                if k not in ret_dict:
                    ret_dict[k] = []
                ret_dict[k].append(ret[k])
        ret_dict = {k : torch.cat(ret_dict[k], 0) for k in ret_dict}

        for k in ret_dict:
            k_sh = list(sh[:-1]) + list(ret_dict[k].shape[1:])
            ret_dict[k] = torch.reshape(ret_dict[k], k_sh)

        return ret_dict

    def batch_render_test(self, test_poses, chunk, 
        render_kwargs, normal_mode=False, img_id=0, light_param=None):
        ''' Render with a batch of testing poses.
        Args:
            test_poses: [batch_sizes, 3, 5]. Testing C2W Matrics.
            chunk: int. Maximum number of rays to process simultaneously. Used to
                control maximum memory usage. Does not affect final results.
            normal_mode: bool. If True, use normal_mode. 
            img_id: which embedding vector to test relighting.
            light_param: input lighting parameters. (See train.py)
        Returns:
            rgb_map: [batch_size, 3]. Predicted RGB values for rays.
            disp_map: [batch_size]. Disparity map. Inverse of depth.
            acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
            ...
        '''
        ret_dict = {}
        if test_poses is None:
            test_poses = [None]
        else:
            if test_poses.ndim == 2:
                test_poses = test_poses[None,...]
            elif self.args.verbose: 
                test_poses = tqdm(test_poses)

        for i, pose in enumerate(test_poses):
            #print(i, time.time() - t)
            #t = time.time()
            result_dict = self(chunk=chunk, test_pose=pose, img_id=img_id, testing=True, 
                    normal_mode=normal_mode, light_param=light_param, **render_kwargs)
            for k in result_dict:
                if k not in ret_dict:
                    ret_dict[k] = []
                ret_dict[k].append(result_dict[k].cpu().numpy())

        ret_dict = {k : np.stack(ret_dict[k], 0) for k in ret_dict}
        return ret_dict

    def calc_normal_grid(self, device, chunk, bbox=None):
        """Calculate a 3D grid of estimated normal
        Args:
            device: device of the model (CPU or CUDA).
            chunk: int. Maximum number of rays to process simultaneously. Used to
                control maximum memory usage. Does not affect final results.
            bbox: [2, 3]. Bounding box of the object.
        Returns:
        """

        # If no bounding box is given, initialize a (enough large) bounding box
        if bbox is None:
            bbox = torch.zeros([2, 3], device=device)
            bbox[0] = bbox[0]-3
            bbox[1] = bbox[1]+3

        # Tighten the bounding box
        res = 64
        while res < 512: 
            print(res, bbox)
            t = torch.arange(0, res, device=device)
            x, y, z = torch.meshgrid(t, t, t)
            coord = torch.stack([x,y,z], dim=-1).reshape(-1, 3) # res^3, 3
            coord = (bbox[1]-bbox[0])/(res-1)*coord+bbox[0]

            # find voxels where the density is larger than zero
            sigma = []
            rays_range = list(range(0, coord.shape[0], chunk))
            if self.args.verbose:
                rays_range = tqdm(rays_range)
            for i in rays_range:
                sigma.append(self._render_sigma(coord[i:i+chunk]))

            sigma = torch.cat(sigma, dim=0).reshape(res, res, res)
            valid_coord = torch.nonzero(sigma>0.01)
            assert(valid_coord.sum()>0)

            # update bounding box
            bbox_coord = torch.stack([valid_coord.min(dim=0)[0], valid_coord.max(dim=0)[0]], dim=0) # 2, 3
            bbox_new = (bbox[1]-bbox[0])/(res-1)*(bbox_coord-(res-1)//2)*1.2+(bbox[0]+bbox[1])/2 # 2, 3
            bbox[0] = torch.maximum(bbox[0], bbox_new[0])
            bbox[1] = torch.minimum(bbox[1], bbox_new[1])
            res *= 2

        ### Normal Extraction


        # Remapping Function
        normal_alpha = self.args.normal_smooth_alpha
        sigma = -(1/normal_alpha)*torch.exp(-normal_alpha*sigma[None,None,:,:,:][:,[0,0,0],:,:,:]) # 1 x 3 x t x t x t

        # Convolution with Sobel Kernel
        kernel_size = 5
        kernel_range = torch.arange(kernel_size, device=device)
        kernel_coord = torch.stack(torch.meshgrid(kernel_range, kernel_range, kernel_range), dim=0).float()-kernel_size//2 # 3 x k x k x k
        kernel_coord_norm = (torch.norm(kernel_coord, dim=0, keepdim=True)+1e-9)
        kernel = -kernel_coord / kernel_coord_norm / kernel_coord_norm # 3 x k x k x k

        kernel = kernel[None,:,:,:,:][[0,0,0],:,:,:,:] # 3 x 3 x t x t x t
        kernel *= torch.eye(3, device=kernel.device)[:,:,None,None,None]
        self.normal_grid = F.conv3d(sigma, kernel, padding=kernel_size//2) # 1 x 3 x t x t x t
        self.normal_grid = self.normal_grid[0].permute(1,2,3,0)
        self.bbox = bbox

    def calc_normal(self, pixels, img_id, chunk, near=0., far=1., **kwargs):
        """Calculate normal given a batch of ray pixels.
        Args:
            pixels: [2, N_rays, 3]. Pixel coordinates of the rays.
            img_id: [N_rays]. Image index of the rays.              
            chunk: int. Maximum number of rays to process simultaneously. Used to
            control maximum memory usage. Does not affect final results.         
            near: [batch_size]. Nearest distance for a ray.
            far: [batch_size]. Farthest distance for a ray.
        Returns:
            normal: [N_rays, 3]. The normal supervision of the rays.
        """

        assert(self.normal_grid is not None), "Need to run calc_normal_grid first!"

        rays_range = list(range(0, pixels.shape[1], chunk))
        if self.args.verbose:
            rays_range = tqdm(rays_range)
        normal_list = []
        for i in rays_range:
            rays_o, rays_d = self.pix2rays(pixels[:,i:i+chunk], img_id[i:i+chunk])

            rays_o = torch.reshape(rays_o, [-1,3]).float()
            rays_d = torch.reshape(rays_d, [-1,3]).float()
            img_id_chunk = torch.reshape(img_id[i:i+chunk], [-1,1]).float()

            near_d, far_d = self.get_depth_range(rays_o, rays_d, self.bbox.T.cpu().numpy())
            near_d = torch.clamp(near_d, near)[:,None]
            far_d = torch.clamp(far_d, 0, far)[:,None]
            rays = torch.cat([rays_o, rays_d, near_d, far_d], -1)
            rays = torch.cat([rays, img_id_chunk], -1)

            rays = torch.cat([rays, rays*0], -1)

            # Get sample points
            ret = self._render_rays(rays, testing=False, **kwargs)
            weights = ret['static_only_weights'] 
            pts = ret['z_vals'][:,:,None]*rays_d[:,None,:]+rays_o[:,None,:]

            # Retrieve normal from the pre-computed normal grid (trilinear interpolation)
            pts_shifted = (pts-self.bbox[None,None,0])/(self.bbox[1]-self.bbox[0]+1e-9) # N_ray, N_sample, 3
            coord = pts_shifted.clip(0, 1)*2-1 # N_ray, N_sample, 3
            normals = F.grid_sample(self.normal_grid.permute(3,0,1,2)[None,:,:,:,:], coord[None,None,:,:,[2,1,0]], mode='bilinear', align_corners=True) # 1, 3, 1, N_ray, N_sample
            normals = normals[0,:,0,:,:].permute(1, 2, 0) # N_ray, N_sample, 3
            normals = (weights[:,:,None] * normals).sum(dim=1) # / ((weights[:,:,None] * normals).norm(dim=-1, keepdim=True).sum(dim=1)+1e-9)
            normals = normals / normals.norm(dim=-1, keepdim=True).clip(1) * (weights.sum(dim=1, keepdim=True)>0.1)
            normal_list.append(normals.detach())

        return torch.cat(normal_list, dim=0)
