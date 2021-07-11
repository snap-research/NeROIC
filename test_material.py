import os, sys
import numpy as np
import imageio
import json
import random
import time
from pytorch_lightning.utilities.distributed import rank_zero_only
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim.lr_scheduler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning import loggers as pl_loggers

from opt import config_parser

from dataset.llff import LLFFDataset
from models.neroic_renderer import NeROICRenderer
import models.network.neroic as neroic

from utils.utils import *
from utils import exposure_helper
import pickle
import models.sh_functions as sh

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

class NeRFSystem(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        args.N_rand = 30000000000
        self.args = args

        if args.model == 'NeROIC':
            self.renderer = NeROICRenderer(args)
        else:
            raise ValueError("Unsupported model.")

        self.basedir = args.basedir
        self.expname = args.expname

        self.render_kwargs_test = {
            'perturb' : args.perturb,
            'N_importance' : args.N_importance,
            'N_samples' : args.N_samples,
            'use_viewdirs' : args.use_viewdirs,
            'raw_noise_std' : args.raw_noise_std,
        }

        self.render_kwargs_test['lindisp'] = args.lindisp
        self.render_kwargs_test['perturb'] = False
        self.render_kwargs_test['N_samples'] = self.render_kwargs_test['N_samples']*4
        self.render_kwargs_test['raw_noise_std'] = 0.

    def forward(self, pixel_coords, pose, img_id): # Rendering 
        return self.renderer(pixel_coords=pixel_coords, test_pose=pose, img_id=img_id, 
                                chunk=self.args.chunk, **self.render_kwargs_train)

    def test_step(self, batch, batch_idx):
        hwf = batch['poses'][0][:3,-1]
        img_id = batch['img_id'][0]
        print(img_id, batch_idx)
        pose = self.renderer.get_pose(img_id, hwf)
        ret_dict = self.renderer.batch_render_test(pose, self.args.chunk//4, self.render_kwargs_test, img_id=img_id)

        gt_imgs = batch['gt_color'][0]
        gt_masks = batch['gt_mask'][0]

        rgbs = torch.FloatTensor(ret_dict['rgb_map']).to(gt_imgs.device)
        if 'static_only_acc_map' in ret_dict:
            rgbs_acc = torch.FloatTensor(ret_dict['static_only_acc_map']).to(gt_imgs.device)[...,[0,0,0]]
        else:
            rgbs_acc = torch.FloatTensor(ret_dict['acc_map']).to(gt_imgs.device)[...,None][...,[0,0,0]]
        rgbs_coarse = torch.FloatTensor(ret_dict['rgb_map_coarse']).to(gt_imgs.device)
        rgbs_static = torch.FloatTensor(ret_dict['static_rgb_map']).to(gt_imgs.device)

        if 'albedo_map' in ret_dict: # albedo map
            rgbs_albedo = torch.FloatTensor(ret_dict['albedo_map']).to(gt_imgs.device)
        else:
            rgbs_albedo = rgbs
        if 'spec_map' in ret_dict: # specular map
            rgbs_specular = torch.FloatTensor(ret_dict['spec_map']).to(gt_imgs.device)
        else:
            rgbs_specular = rgbs
        if 'glossiness_map' in ret_dict: # glossiness map
            rgbs_glossiness = torch.FloatTensor(ret_dict['glossiness_map']).to(gt_imgs.device)
        else:
            rgbs_glossiness = rgbs
        if 'transient_acc_map' in ret_dict: # transient accumulation map
            rgbs_transient = torch.FloatTensor(ret_dict['transient_acc_map']).to(gt_imgs.device)[...,None][...,[0,0,0]]
        else:
            rgbs_transient = rgbs
        if 'is_edge' in ret_dict: # edge map
            rgbs_is_edge = torch.FloatTensor(ret_dict['is_edge']).to(gt_imgs.device)[...,None][...,[0,0,0]]
        else:
            rgbs_is_edge = rgbs
        
        if self.args.model_type == "rendering": # sh env lighting map
            rgbs_light = sh.unproject_environment(3, self.renderer.env_lights[img_id], 
                            rgbs.shape[1], rgbs.shape[2])
        else:
            rgbs_light = rgbs[0]

        gt_imgs = gt_imgs
        if self.args.debug_green_bkgd:
            bkgd = torch.from_numpy(np.array([0,1,0])).type_as(rgbs)   
        else:
            bkgd = torch.from_numpy(np.array([1,1,1])).type_as(rgbs)

        log = {}
        img_loss = img2mse(rgbs, gt_imgs*gt_masks[...,None] + bkgd*(~gt_masks[...,None]))
        loss = img_loss	      
        psnr = mse2psnr(img_loss)	
        log = {'val_loss': loss, 'val_psnr': psnr}	      

        img = rgbs[0].clamp(0, 1).cpu() # (H, W, 3)
        img_static = rgbs_static[0].clamp(0, 1).cpu() # (H, W, 3)
        img_albedo = rgbs_albedo[0].clamp(0, 1).cpu() # (H, W, 3)
        img_specular = rgbs_specular[0].clamp(0, 1).cpu() # (H, W, 3)
        img_glossiness = rgbs_glossiness[0].clamp(0, 1).cpu() # (H, W, 3)
        img_transient = rgbs_transient[0].clamp(0, 1).cpu() # (H, W, 3)
        img_is_edge = rgbs_is_edge[0].clamp(0, 1).cpu() # (H, W, 3)
        img_acc = rgbs_acc[0].clamp(0, 1).cpu() # (H, W, 3)
        img_light = rgbs_light.clamp(0, 1).cpu() # (H, W, 3)
        print(img_transient.max())

        img_gt = gt_imgs.cpu() # (H, W, 3)
        depth = visualize_depth(ret_dict['depth_map'][0], cmap=cv2.COLORMAP_HOT).permute(1, 2, 0) # (H, W, 3)

        if 'normal_map_weighted' in ret_dict:
            rot_mat = self.renderer.get_rotation(img_id)   
            normal = torch.FloatTensor(ret_dict['normal_map_weighted'][0]).type_as(gt_imgs).reshape(-1, 3).T # 3 x HW
            normal = torch.matmul(torch.inverse(rot_mat),normal).T.reshape(gt_imgs.shape)
            normal = (normal+1)/2
            normal = normal.clamp(0, 1).cpu() # (H, W, 3)
        else:
            normal = img

        gt_masks = gt_masks.cpu()
        img_acc = img_acc.cpu()
        def mto8b(image, color=bkgd.cpu()):
            return to8b((image.cpu()*img_acc + color*(1-img_acc)).numpy())
        
        imageio.imwrite(os.path.join(self.logger.save_dir, self.args.expname, "%d_gt.png"%batch_idx), to8b(img_gt.cpu().numpy()))
        imageio.imwrite(os.path.join(self.logger.save_dir, self.args.expname, "%d.png"%batch_idx), mto8b(img))
        imageio.imwrite(os.path.join(self.logger.save_dir, self.args.expname, "%d_depth.png"%batch_idx), depth)
        imageio.imwrite(os.path.join(self.logger.save_dir, self.args.expname, "%d_static.png"%batch_idx), mto8b(img_static))
        imageio.imwrite(os.path.join(self.logger.save_dir, self.args.expname, "%d_albedo.png"%batch_idx), mto8b(img_albedo))
        imageio.imwrite(os.path.join(self.logger.save_dir, self.args.expname, "%d_specular.png"%batch_idx), mto8b(img_specular))
        imageio.imwrite(os.path.join(self.logger.save_dir, self.args.expname, "%d_glossiness.png"%batch_idx), mto8b(img_glossiness))
        imageio.imwrite(os.path.join(self.logger.save_dir, self.args.expname, "%d_transient.png"%batch_idx), to8b(img_transient.cpu().numpy()))
        imageio.imwrite(os.path.join(self.logger.save_dir, self.args.expname, "%d_edge.png"%batch_idx), to8b(img_is_edge.cpu().numpy()))
        imageio.imwrite(os.path.join(self.logger.save_dir, self.args.expname, "%d_light.png"%batch_idx), to8b(img_light.cpu().numpy()))
        imageio.imwrite(os.path.join(self.logger.save_dir, self.args.expname, "%d_normal.png"%batch_idx), mto8b(normal))

        return log


    def test_epoch_end(self, outputs):
        pass

    def setup(self, stage):
        self.args.split = self.args.test_split
        if self.args.dataset_type == 'llff':
            self.test_dataset = LLFFDataset(self.args, recenter=True, bd_factor=0.75, path_zflat=False)        
        else:
            raise ValueError('Unknown dataset type: %s'%self.args.dataset_type)

        self.bds_dict = {
            'near' : self.test_dataset.near,
            'far' : self.test_dataset.far,
            'bbox': self.test_dataset.bbox,
        }
        self.render_kwargs_test.update(self.bds_dict)
        self.renderer.init_cam_pose(self.test_dataset.get_all_poses())

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=True)

def train():

    parser = config_parser()
    args = parser.parse_args()

    args.verbose = True
    args.have_mask = True # enforce bg/fg mask
    args.mask_ratio = 100000
    args.debug_green_bkgd = False

    logger = pl_loggers.TensorBoardLogger(
        save_dir="results/material",
        name=args.expname
    )

    # Create log dir and copy the config file
    nerf_sys = NeRFSystem.load_from_checkpoint(checkpoint_path=args.ft_path, map_location=None, **{'args': args}, strict=False)

    trainer = Trainer(gpus=1, logger=logger)
    trainer.test(nerf_sys)


if __name__=='__main__':
    train()

