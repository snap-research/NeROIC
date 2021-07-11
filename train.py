import os, sys
import numpy as np
import imageio
import json
import random
import time
from pytorch_lightning.accelerators import accelerator
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
import models.sh_functions as sh

from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

class NeROICSystem(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.model == 'NeROIC':
            self.renderer = NeROICRenderer(args)
        else:
            raise ValueError("Unsupported model.")

        self.basedir = args.basedir
        self.expname = args.expname
        self.model_type = args.model_type

        self.render_kwargs_train = {
            'perturb' : args.perturb,
            'N_importance' : args.N_importance,
            'N_samples' : args.N_samples,
            'use_viewdirs' : args.use_viewdirs,
            'raw_noise_std' : args.raw_noise_std,
        }

        # NDC only good for LLFF-style forward facing data
        # Since this model is designed for 360 images, NDC is not supported here
        # self.render_kwargs_train['ndc'] = False
        self.render_kwargs_train['lindisp'] = args.lindisp

        self.render_kwargs_test = {k : self.render_kwargs_train[k] for k in self.render_kwargs_train}
        self.render_kwargs_test['perturb'] = False
        self.render_kwargs_test['N_samples'] = self.render_kwargs_test['N_samples']*4 # more samples during testing
        self.render_kwargs_test['raw_noise_std'] = 0.

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(params=self.renderer.parameters(), lr=self.args.lrate, eps=1e-8, weight_decay=0)#betas=(0.9, 0.999))
        if self.args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.decay_epoch[0], eta_min=1e-6)
        elif self.args.scheduler == "multistep":
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.decay_epoch[0], gamma=self.args.decay_gamma)
        return [self.optimizer], [scheduler]

    def forward(self, pixel_coords, pose, img_id): # Rendering 
        return self.renderer(pixel_coords=pixel_coords, test_pose=pose, img_id=img_id, 
                                chunk=self.args.chunk, **self.render_kwargs_train)

    def training_step(self, batch, batch_idx):    
        rays = batch['rays'][0]
        pose = batch['poses'][0]
        img_id = batch['img_id'][0]

        gt_color = batch['gt_color'][0]
        rays_is_bg = batch['rays_is_bg'][0]

        gt_normal = batch['gt_normal'][0]

        ret_dict = self.forward(rays, pose, img_id)
        loss_dict, prog_list = self.renderer.calculate_loss(ret_dict, gt_color, gt_normal, rays_is_bg, 
                                                            self.bds_dict, img_id)
        
        # Rendering from testing views during training
        if self.trainer.is_global_zero:
            is_video_step = (self.global_step%self.args.i_video==0 and self.global_step > 0)
            is_traintesting_step = (self.global_step%self.args.i_traintest==0 and self.global_step > 0) 
            is_last_epoch = (batch_idx==0 and self.current_epoch == self.args.num_epochs-1)

            if is_video_step or is_last_epoch:      
                movie_dir = '{}/{}_{:06d}'.format(self.logger.log_dir, self.expname, self.global_step)
                os.makedirs(movie_dir, exist_ok=True)
                moviebase = os.path.join(movie_dir, '{}_spiral_{:06d}_'.format(self.expname, self.global_step))
                
                poses = self.train_dataset.get_test_poses().to(rays.device)
                ret_dict_list = []

                # Turn on testing mode          
                with torch.no_grad():
                    for i in trange(len(poses)):
                        ret_dict_list.append(self.renderer.batch_render_test(poses[i:i+1,...], 
                            self.args.chunk//2, self.render_kwargs_test, 
                            img_id=self.args.test_img_id))
                        imageio.imwrite(os.path.join(movie_dir, "%02d.png"%i), 
                            to8b(ret_dict_list[-1]['static_rgb_map'][0]))

                # Saving output buffers
                for k in ret_dict_list[0].keys():
                    v = np.concatenate([x[k] for x in ret_dict_list], axis=0)
                    if k == 'depth_map':
                        v = (visualize_depth(v)*255).cpu().numpy().transpose([0,2,3,1]).astype(np.uint8)
                    elif k == 'normal_map_weighted':
                        origin_shape = v.shape
                        v = torch.from_numpy(v).type_as(poses).reshape(poses.shape[0], -1, 3).transpose(1, 2) # B x 3 x HW
                        v = torch.bmm(torch.inverse(poses[:,:3,:3]),v).transpose(1, 2).reshape(origin_shape).cpu().numpy()
                        v = to8b((v+1)/2)
                    else:
                        v = to8b(v)
                    imageio.mimwrite(moviebase + '%s.mp4'%k, v, fps=30, quality=8)     

            # Rendering from training poses during training
            if is_traintesting_step or is_last_epoch:        
                movie_dir = '{}/{}_{:06d}'.format(self.logger.log_dir, self.expname, self.global_step)
                os.makedirs(movie_dir, exist_ok=True)

                poses = self.train_dataset.get_train_poses().to(rays.device)
                ret_dict_list = []

                # Turn on testing mode        
                with torch.no_grad():
                    for i in trange(min(5, len(poses))):
                        ret_dict_list.append(self.renderer.batch_render_test(poses[i:i+1,...], 
                            self.args.chunk//2, self.render_kwargs_test,
                            img_id=self.train_dataset.i_train[i]))
                        imageio.imwrite(os.path.join(movie_dir, "train_%02d_static.png"%i), to8b(ret_dict_list[-1]['static_rgb_map'][0]))
                        imageio.imwrite(os.path.join(movie_dir, "train_%02d.png"%i), to8b(ret_dict_list[-1]['rgb_map'][0]))

        for loss_key, loss_val in loss_dict.items():
            prog = (loss_key in prog_list)
            self.log('train_%s'%loss_key, loss_val, prog_bar=prog, rank_zero_only=True)
        self.log('lr', get_learning_rate(self.optimizer), rank_zero_only=True)

        return loss_dict['loss']
        
    def validation_step(self, batch, batch_idx):

        poses = batch['poses']
        gt_imgs = batch['gt_color']
        gt_masks = batch['gt_mask']
    
        ret_dict = self.renderer.batch_render_test(poses, self.args.chunk//4, self.render_kwargs_test, img_id=self.args.test_img_id)

        rgbs = torch.FloatTensor(ret_dict['rgb_map']).to(gt_imgs.device)
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
            rgbs_light = sh.unproject_environment(3, self.renderer.env_lights[self.args.test_img_id], 
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

        if self.trainer.is_global_zero and batch_idx == 0:        

            img = rgbs[0].clamp(0, 1).permute(2, 0, 1).cpu() # (3, H, W)
            img_coarse = rgbs_coarse[0].clamp(0, 1).permute(2, 0, 1).cpu() # (3, H, W)
            img_static = rgbs_static[0].clamp(0, 1).permute(2, 0, 1).cpu() # (3, H, W)
            img_albedo = rgbs_albedo[0].clamp(0, 1).permute(2, 0, 1).cpu() # (3, H, W)
            img_specular = rgbs_specular[0].clamp(0, 1).permute(2, 0, 1).cpu() # (3, H, W)
            img_glossiness = rgbs_glossiness[0].clamp(0, 1).permute(2, 0, 1).cpu() # (3, H, W)
            img_transient = rgbs_transient[0].clamp(0, 1).permute(2, 0, 1).cpu() # (3, H, W)
            img_is_edge = rgbs_is_edge[0].clamp(0, 1).permute(2, 0, 1).cpu() # (3, H, W)
            img_acc = rgbs_acc[0].clamp(0, 1).permute(2, 0, 1).cpu() # (3, H, W)
            img_light = rgbs_light.clamp(0, 1).permute(2, 0, 1).cpu() # (3, H, W)
            
            img_gt = gt_imgs[0].permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(ret_dict['depth_map'][0]) # (3, H, W)

            if 'normal_map_weighted' in ret_dict:
                normal = torch.FloatTensor(ret_dict['normal_map_weighted'][0]).type_as(gt_imgs).reshape(-1, 3).T # 3 x HW
                normal = torch.matmul(torch.inverse(poses[0,:3,:3]),normal).T.reshape(gt_imgs[0].shape)
                normal = (normal+1)/2
                normal = normal.clamp(0, 1).permute(2, 0, 1).cpu() # (3, H, W)
            else:
                normal = img

            # Validation buffers: gt image, pred image, static-only rgb, transient rgb, coarse rgb,
            #                     depth, albedo, specular, glossiness, normal
            #                     edge, acc map, env lighting
            stack = torch.stack([
            img_gt, img, img_static, img_transient, img_coarse, 
            depth, img_albedo, img_specular, img_glossiness, normal, 
            img_is_edge, img_acc, img_light]) # (4, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth', stack, self.global_step)

        return log

    def setup(self, stage):
        if self.args.dataset_type == 'llff':
            self.args.split = "train"
            self.train_dataset = LLFFDataset(self.args, recenter=True, bd_factor=0.75, path_zflat=False)
            self.args.split = "val"
            self.val_dataset = LLFFDataset(self.args, recenter=True, bd_factor=0.75, path_zflat=False)
        else:
            raise ValueError('Unknown dataset type: %s'%self.args.dataset_type)

        self.bds_dict = {
            'near' : self.train_dataset.near,
            'far' : self.train_dataset.far,
            'bbox' : self.train_dataset.bbox,
        }

        self.render_kwargs_train.update(self.bds_dict)
        self.render_kwargs_test.update(self.bds_dict)
        if self.args.rays_path == "":
            self.train_dataset.generate_rays()
        else: # Load pre-processed rays 
            self.train_dataset.load_rays_from_file(self.args.rays_path)
        self.renderer.init_cam_pose(self.train_dataset.get_all_poses())
        self.train_dataset.print_info()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, shuffle=True, num_workers=4, batch_size=1, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, shuffle=False, num_workers=4, batch_size=1, pin_memory=True)

    def training_epoch_end(self, outputs):
        self.train_dataset.shuffle()
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        # Validation losses do not reflect the quality of the model, as the testing lighting / camera is unknown.
        outputs = self.all_gather(outputs)        
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()	        # mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()	        # mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        self.log('val_loss', mean_loss, prog_bar=False, rank_zero_only=True)	        # self.log('val_loss', mean_loss, prog_bar=False, rank_zero_only=True)
        self.log('val_psnr', mean_psnr, prog_bar=False, rank_zero_only=True)

    def on_train_start(self):
        if self.trainer.is_global_zero:        
            f = os.path.join(self.logger.log_dir, 'args.txt')
            with open(f, 'w') as file:
                for arg in sorted(vars(self.args)):
                    attr = getattr(self.args, arg)
                    file.write('{} = {}\n'.format(arg, attr))
            if self.args.config is not None:
                f = os.path.join(self.logger.log_dir, 'config.txt')
                with open(f, 'w') as file:
                    file.write(open(self.args.config, 'r').read())
        return super().on_train_start()

def train():

    parser = config_parser()
    args = parser.parse_args()

    args.split = "train"

    # Create log dir and copy the config file
    os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
    nerf_sys = NeROICSystem(args)

    # Summary writers
    logger = pl_loggers.TensorBoardLogger(
        save_dir=args.basedir,
        name=args.expname
    )
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.basedir, args.expname), 
                                          filename='{epoch:d}',
                                          monitor=None,
                                          every_n_val_epochs=1)

    if args.load_prior == True: # Train
        nerf_sys = nerf_sys.load_from_checkpoint(args.ft_path, map_location=None, **{'args': args}, strict=False)
    
    trainer = Trainer(
                        max_epochs=args.num_epochs,
                        callbacks=[checkpoint_callback],
                        resume_from_checkpoint= "" if args.load_prior else args.ft_path,
                        logger=logger,
                        weights_summary=None,
                        progress_bar_refresh_rate=1 if args.verbose else 100,
                        gpus=args.num_gpus,
                        accelerator='ddp' if args.num_gpus>1 else None,
                        num_sanity_val_steps=1 if args.verbose else 1,
                        gradient_clip_val=1,
                        benchmark=True,
                        val_check_interval = 1.0 if args.i_testset<=0 else args.i_testset,
                        check_val_every_n_epoch = 1 if args.verbose else args.i_testepoch,
                        profiler="simple" if args.num_gpus==1 else None)

    trainer.fit(nerf_sys)


if __name__=='__main__':
    train()

