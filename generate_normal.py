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
import models.network.neroic as NeROIC

from utils.utils import *

import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

class NeRFSystem(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.model == 'NeROIC':
            self.renderer = NeROICRenderer(args)
        else:
            raise ValueError("Unsupported model.")

        self.basedir = args.basedir
        self.expname = args.expname

        self.render_kwargs_train = {
            'perturb' : args.perturb,
            'N_importance' : args.N_importance,
            'N_samples' : args.N_samples,
            'use_viewdirs' : args.use_viewdirs,
            'raw_noise_std' : args.raw_noise_std,
        }

        self.render_kwargs_train['lindisp'] = args.lindisp

        self.render_kwargs_train['perturb'] = False
        self.render_kwargs_train['N_samples'] = self.render_kwargs_train['N_samples']*4
        self.render_kwargs_train['raw_noise_std'] = 0.

    def forward(self, rays, pose, img_id, normal_mode=True):
        return self.renderer(pixel_coords=rays, test_pose=pose, img_id=img_id, chunk=self.args.chunk//16, normal_mode=normal_mode, **self.render_kwargs_train)

    def test_step(self, batch, batch_idx):
        # TODO: support multiple batches

        rays = batch['rays'][0]
        pose = batch['poses'][0]
        img_id = batch['img_id'][0]

        # Extracting a rough bounding box based on density function (sigma)
        print("Generating Coarse Bounding Box...")
        if not self.args.use_bbox:
            sample_idx = torch.randperm(rays.shape[1])[:rays.shape[1] // 100]
            rays_sample = rays[:, sample_idx, :]
            img_id_sample = img_id[sample_idx]
            ret_dict = self(rays_sample, pose, img_id_sample)
            pts = ret_dict['pts_map'][ret_dict['acc_map']>0.5]
            bbox = enlarge_bbox(torch.stack([pts.min(dim=0)[0], pts.max(dim=0)[0]], dim=0), 1.1)
        else:
            bbox = torch.from_numpy(self.render_kwargs_train['bbox'].T).type_as(rays)

        print("bbox is ", bbox)
        
        print("Generating Normal 3D Grid...")
        self.renderer.calc_normal_grid(rays.device, self.args.chunk, bbox=bbox)
                
        print("Rendering Rays...")
        normal_mean = self.renderer.calc_normal(rays, img_id, 
                            self.args.chunk//8, **self.render_kwargs_train) 

        batch['normal_mean'] = normal_mean[None,...]
        batch['depth_mean'] = normal_mean[None,:,0]
        batch['depth_var'] = normal_mean[None,:,0]

        print("Saving Images for Visualization...")
        for i in range(img_id.max().long()+1):
            pose = self.test_dataset.get_pose(i)
            h, w = pose[0:2, 4].long()
            rays_i = rays[0, img_id[:,0] == i, :2].long()
            if len(rays_i) != h*w:
                continue
            normal_i = normal_mean[img_id[:,0] == i]
            normal_img = torch.zeros_like(normal_i).reshape(h, w, 3)
            normal_img[rays_i[:, 1], rays_i[:, 0]] = normal_i
            normal_img = torch.matmul(torch.inverse(self.renderer.cam_R[i]),normal_img.reshape(-1, 3).T).T.reshape(normal_img.shape)
            normal_img = (normal_img.detach().cpu().numpy()+1)/2
            imageio.imwrite(os.path.join(self.logger.save_dir, self.args.expname, "%d_normal.png"%i), to8b(normal_img))

        print("Saving Rays...")
        if len(self.test_dataset.i_train) != 1: # Safety check
            for k, v in batch.items():
                batch[k] = v.detach().cpu()
            pickle.dump(batch, open(os.path.join(self.logger.save_dir, self.args.expname, "rays.pkl"), "wb"))

    def validation_step(self, batch, batch_idx):
        pass

    def setup(self, stage):
        if self.args.dataset_type == 'llff':
            self.args.split = "train"
            self.test_dataset = LLFFDataset(self.args, recenter=True, bd_factor=0.75, path_zflat=False)        
        elif self.args.dataset_type == 'nerd_real':
            self.args.split = "train"
            self.test_dataset = NerDRealDataset(self.args, recenter=True, bd_factor=0.75, path_zflat=False)
        else:
            raise ValueError('Unknown dataset type: %s'%self.args.dataset_type)

        self.bds_dict = {
            'near' : self.test_dataset.near,
            'far' : self.test_dataset.far,
            'bbox': self.test_dataset.bbox,
        }
        self.render_kwargs_train.update(self.bds_dict)
        self.test_dataset.generate_rays()
        self.renderer.init_cam_pose(self.test_dataset.get_all_poses())
        self.test_dataset.print_info()

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=True)

def train():

    parser = config_parser()
    args = parser.parse_args()

    args.verbose = True
    args.have_mask = True 
    args.mask_ratio = 100000
    # Setting an infinitely large N_rand so the dataloader can load all rays within one batch.
    args.N_rand = 30000000000 

    args.split = "train"

    logger = pl_loggers.TensorBoardLogger(
        save_dir="results/cached_rays",
        name=args.expname
    )

    nerf_sys = NeRFSystem.load_from_checkpoint(checkpoint_path=args.ft_path, map_location=None, **{'args': args}, strict=False)

    trainer = Trainer(gpus=1, logger=logger)
    trainer.test(nerf_sys)


if __name__=='__main__':
    train()

