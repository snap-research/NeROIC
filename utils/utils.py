import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.transforms as T
import cv2
from PIL import Image

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2bce = lambda x, y : F.binary_cross_entropy(x, y)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
import torch
from kornia.losses import ssim as dssim

def mse2ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 3, reduction) # dissimilarity in [0, 1]
    return 1-2*dssim_ # in [-1, 1]
    
def normalize(tensor):
    return tensor / (tensor.norm(dim=-1, keepdim=True)+1e-9)

def enlarge_bbox(bbox, s):
    mid = (bbox[0]+bbox[1])/2
    diff = (bbox[1]-bbox[0])/2
    bbox[0] = mid-diff*s
    bbox[1] = mid+diff*s
    return bbox

def isclose(x, val, threshold = 1e-6):
    return torch.abs(x - val) <= threshold

def safe_sqrt(x):
    sqrt_in = torch.relu(torch.where(isclose(x, 0.0), torch.ones_like(x) * 1e-6, x))
    return torch.sqrt(sqrt_in)

def safe_pow(x, p):
    sqrt_in = torch.relu(torch.where(isclose(x, 0.0), torch.ones_like(x) * 1e-6, x))
    return torch.pow(sqrt_in, p)
    
def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def hwf2mat(hwf):
    H, W, focal = hwf
    H, W = int(H), int(W)
    return np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]])

def batched_hwf2mat(hwf):
    H = hwf[:, 0].astype(np.int32)
    W = hwf[:, 1].astype(np.int32)
    focal = hwf[:, 2]
    K = np.zeros([hwf.shape[0], 3, 3], dtype=np.float32)
    K[:, 0, 0] = focal 
    K[:, 1, 1] = focal
    K[:, 0, 2] = W*0.5
    K[:, 1, 2] = H*0.5
    return K

def get_rays(pose, mask=None):
    if isinstance(pose, np.ndarray):
        pose = torch.FloatTensor(pose)
    H = int(pose[0, 4])
    W = int(pose[1, 4])
    K = hwf2mat(pose[:3, 4])
    c2w = pose[:3, :4]
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    if mask is None:
        i = i.t().to(pose.device)
        j = j.t().to(pose.device)
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3,-1].expand(rays_d.shape)
        return torch.stack([rays_o, rays_d], dim=0)
    else:
        i = i.t().to(pose.device)
        j = j.t().to(pose.device)
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3,-1].expand(rays_d.shape)
        mask_idx = torch.nonzero(mask)
        rays_d = rays_d[mask_idx[:,0], mask_idx[:,1]]
        rays_o = rays_o[mask_idx[:,0], mask_idx[:,1]]
        return torch.stack([rays_o, rays_d], dim=0)
    
def get_pixels(pose, H=None, W=None, mask=None):
    if isinstance(pose, np.ndarray):
        pose = torch.FloatTensor(pose)
    if H == None:
        H = int(pose[0, 4])
        W = int(pose[1, 4])
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H)) # H x W # pytorch's meshgrid has indexing='ij'
    i = i.t().to(pose.device)
    j = j.t().to(pose.device)
    px = torch.zeros_like(i[:,:,None]).repeat(1, 1, 3)
    px[:,:,0] = i
    px[:,:,1] = j
    return torch.stack([px, torch.zeros_like(px)], dim=0) # 2 x H x W x 3


def get_rays_np(pose, mask=None):
    rays_o, rays_d = get_rays(pose, mask)
    return rays_o.cpu().detach().numpy(), rays_d.cpu().detach().numpy()

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d

def visualize_depth_disp(disp, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = np.nan_to_num(disp) # change nan to 0
    x = (255*np.clip(x/10.0, 0, 1)).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (B, H, W)
    """
    non_batch=False
    if depth.ndim == 2:
        non_batch=True
        depth = depth[None,:,:]
    ret_list = []
    for i in range(len(depth)):
        x = np.nan_to_num(depth[i]) # change nan to 0
        mi = np.min(x[np.nonzero(x)]) # get minimum depth
        ma = np.max(x)
        x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
        x = 1-x.clip(0, 1)
        x = (255*x).astype(np.uint8)
        x_ = Image.fromarray(cv2.applyColorMap(x, cmap)[...,[2,1,0]])
        
        ret_list.append(T.ToTensor()(x_)) # (3, H, W)
    ret_list = torch.stack(ret_list, dim=0)
    if non_batch:
        ret_list = ret_list[0]
    return ret_list
    