from math import pi
import os, sys
from unicodedata import normalize
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.autograd as autograd
from tqdm import tqdm, trange

import utils.utils as utils
from models.base_render import *
import models.network 
import models.network.neroic as neroic
import models.sh_functions as sh

from einops import repeat

class NeROICRenderer(BaseRenderer):
    def __init__(self, args):
        super(NeROICRenderer, self).__init__(args)        

        self.sh_order = 3 
        self.model_type = args.model_type

        # build embeddings 
        embed_fn, input_ch = neroic.get_embedder(args.multires, args.i_embed)
        input_ch_views = 0
        embeddirs_fn = None
        embedding_t = None
        embedding_a = None

        if args.use_viewdirs:
            embeddirs_fn, input_ch_views = neroic.get_embedder(args.multires_views, args.i_embed)
        if args.encode_transient:
            embedding_t = torch.nn.Embedding(args.N_vocab, args.N_tau)
        if args.encode_appearance:
            embedding_a = torch.nn.Embedding(args.N_vocab, args.N_a)

        self.embeds = {
            "embed_fn": embed_fn,
            "embeddirs_fn": embeddirs_fn,
        }

        # build models 
        skips = [4]
        model_coarse = neroic.NeROIC(args, model_type="geometry", 
                            D=args.netdepth, W=args.netwidth, skips=skips, 
                            in_channels_xyz=input_ch, in_channels_dir=input_ch_views,
                            encode_appearance=False, encode_transient=False)
        model_fine = None
        if args.N_importance > 0:
            model_fine =  neroic.NeROIC(args, model_type=args.model_type, 
                                D=args.netdepth, W=args.netwidth, skips=skips, 
                                in_channels_xyz=input_ch, in_channels_dir=input_ch_views,
                                in_channels_a=args.N_a, in_channels_t=args.N_tau,
                                encode_appearance=args.encode_appearance, 
                                encode_transient=args.encode_transient, 
                                beta_min=args.beta_min)
        

        if args.model_type == "rendering":
            init_light = torch.zeros([args.N_vocab, (self.sh_order+1)*(self.sh_order+1), 3])
            
            # initialize lighting with a white ambient light
            init_light[:, 0, :] = 3
            self.env_lights = nn.Parameter(init_light, requires_grad=True)

            # pre-process the coefficients for fast SH intergration
            pow_num, s = sh.pre_calc_sh_mat()
            self.sh_pow_num = nn.Parameter(pow_num, requires_grad=False)
            self.sh_s = nn.Parameter(s, requires_grad=False)

            self.gamma = nn.Parameter(torch.ones([args.N_vocab])*2.4, requires_grad=True)
        else:
            self.env_lights = None
            self.gamma = None

        # all trainable functions
        self.networks = nn.ModuleDict({
            "model_geom_coarse": model_coarse,
            "model_geom_fine": model_fine,
            "embed_t": embedding_t,
            "embed_a": embedding_a,
        })

    def _inference(self, network, pts, viewdirs, embedded_a=None, embedded_t=None, freeze_geometry=False):
        '''Generate raw outputs from the neural network.
        Args:
            network: neural network model.
            pts: [N_rays, N_samples, 3]. sampled points' positions. 
            viewdirs: [N_rays, N_samples, 3]. sampled points' viewing directions. 
            embedded_a: [N_rays, D_a]. appearance embedding vector.
            embedded_t: [N_rays, D_t]. transient embedding vector.
            freeze_geometry: whether to freeze the geometry model (do not pass gradient).
        Returns: 
            raw: [N_rays, N_samples, N_raw]. Raw output from the neural network.
        '''
        N_rays = pts.shape[0]
        N_samples = pts.shape[1]
        chunk = self.netchunk
        output_transient = (embedded_t is not None)
        B = N_rays*N_samples
        out_chunks = []

        pts_flat = pts.reshape(-1, 3)
        inputs = [self.embeds['embed_fn'](pts_flat)]

        # Perform model inference to get rgb and raw sigma
        if viewdirs is not None: 
            viewdirs_flat = repeat(viewdirs, 'n1 c -> (n1 n2) c', n2=N_samples)
            inputs += [self.embeds['embeddirs_fn'](viewdirs_flat)]
        # create other necessary inputs
        if embedded_a is not None:
            inputs += [repeat(embedded_a, 'n1 c -> (n1 n2) c', n2=N_samples)]
        if output_transient:
            inputs += [repeat(embedded_t, 'n1 c -> (n1 n2) c', n2=N_samples)]
        inputs = torch.cat(inputs, 1)
        
        for i in range(0, B, chunk):
            out_chunks += [network(inputs[i:i+chunk], output_transient=output_transient, freeze_geometry=freeze_geometry)]

        out = torch.cat(out_chunks, 0)
        out = out.reshape(N_rays, N_samples, -1)
        return out

    def raw2outputs(self, model_type, raw, rays_d, z_vals, env_light, raw_noise_std=0, out_of_box=None,
                    output_transient=False, testing=False, geom_only=False, geom_cache=None, gamma=2.4):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            model_type: Str. Type of the model (geometry or rendering).
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
            env_light: [num_rays, n_SH, 3]. SH coefficient of environment light.
            gamma: [num_rays]. Gamma values for tone-mapping.
            out_of_box: [num_rays, 1]. Whether the ray is outside of the bounding box.
            testing: whether in testing mode.
            geom_only: whether predict geometry outputs only.
            geom_cache: stored geometry cache from previous passes (for speedup)
        Returns:
            Geometry Components:
                weights: the weight of each sample point
                acc_map: accumulated weight
                static_acc_map: static accumulated weight
                static_only_acc_map: static accumulated weight w/o the effect of transient part
                transient_acc_map: transient accumulated weight
                depth_map: expected depth of each ray
                depth_var_map: variance of depth of each ray
                transient_sigmas: density of transient object
                beta: uncertainty of transient object
            
            Rendering Component:
                rgb_map: rendered rgb color
                normal_map: normal of each ray
                normal_map_weighted: normal map multiplied with accumulated weight (only for visualization)
                spec_map: specular fraction of the rendered result
                static_rgb_map: rendered rgb color with static object only 
                albedo_map: albedo of the object            
            ...
        """
        results = {} if geom_cache is None else geom_cache
        rendering_mode = (model_type=="rendering")
        color_dim = 8 if rendering_mode else 3

        if out_of_box is None:
            out_of_box = (z_vals[:, 0] <= 0)[:,None].float() # invalid rays will be assigned a negative z_val
        
        if self.args.debug_green_bkgd:
            bkgd = torch.from_numpy(np.array([1-testing,1,1-testing])).type_as(raw)   
        else:
            bkgd = torch.from_numpy(np.array([1,1,1])).type_as(raw)   

        
        ### Calculate Static/Transient Geometry and Transient Color
        if geom_cache is None: # if there's no geometry prior presented, calculate sigma from scratch
            results['z_vals'] = z_vals

            static_sigmas = raw[..., color_dim] # N_rays, N_samples_
            if output_transient:
                transient_rgbs = raw[..., color_dim+1:color_dim+4] # N_rays, N_samples_, 3
                transient_sigmas = raw[..., color_dim+4] # N_rays, N_samples_
                transient_betas = raw[..., color_dim+5] # N_rays, N_samples_
            else:
                static_betas = raw[..., color_dim+1] # N_rays, N_samples_

            # Convert these values using volume rendering
            deltas = (z_vals[:, 1:] - z_vals[:, :-1]).clamp(0) # N_rays, N_samples_-1
            delta_inf = 1e2 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
            deltas = torch.cat([deltas, delta_inf], -1)  # N_rays, N_samples_
    
            if output_transient:
                if self.args.transient_lerp_mode:
                    alphas = 1-torch.exp(-deltas*static_sigmas)
                    static_alphas = alphas
                    transient_alphas = alphas.detach()
                else:
                    static_alphas = 1-torch.exp(-deltas*static_sigmas)
                    transient_alphas = 1-torch.exp(-deltas*transient_sigmas) 
                    alphas = 1-torch.exp(-deltas*(static_sigmas+transient_sigmas))
            else:
                noise = torch.randn_like(static_sigmas) * raw_noise_std
                alphas = 1-torch.exp(-deltas*torch.relu(static_sigmas+noise))
                static_alphas = alphas*0
                transient_alphas = alphas*0

            alphas *= 1-out_of_box
            static_alphas *= 1-out_of_box
            transient_alphas *= 1-out_of_box

            alphas_shifted = \
                torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas], -1) # [1, 1-a1, 1-a2, ...]
            transmittance = torch.cumprod(alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]

            weights = alphas * transmittance # N_rays, N_samples
            weights_sum = weights.sum(dim=-1) # N_rays
            results[f'weights'] = weights
            results[f'acc_map'] = weights_sum

            if output_transient:
                if self.args.transient_lerp_mode:
                    static_weights = static_alphas * transmittance            
                    results[f'static_weights'] = static_weights
                    results[f'static_acc_map'] = static_weights.sum(dim=-1)

                    transient_weights = 1-torch.exp(-transient_sigmas)
                    results['transient_sigmas'] = transient_sigmas
                    results[f'transient_acc_map'] = transient_weights.sum(dim=-1) 

                    results['beta'] = (transient_weights*transient_betas).sum(dim=1) # N_rays
                    # Add beta_min AFTER the beta composition. Different from eq 10~12 in the paper.
                    # See "Notes on differences with the paper" in README.
                    results['beta'] += self.args.beta_min

                    static_only_weights = static_weights
                    static_only_weights_sum = static_only_weights.sum(dim=1, keepdim=True) 
                    results[f'static_only_weights'] = static_only_weights    
                    results[f'static_only_acc_map'] = static_only_weights_sum    
                else:
                    static_weights = static_alphas * transmittance
                    results[f'static_weights'] = static_weights
                    results[f'static_acc_map'] = static_weights.sum(dim=-1)

                    transient_weights = transient_alphas * transmittance
                    results[f'transient_acc_map'] = transient_weights.sum(dim=-1)

                    results['beta'] = (transient_weights*transient_betas).sum(dim=1) # N_rays
                    # Add beta_min AFTER the beta composition. Different from eq 10~12 in the paper.
                    # See "Notes on differences with the paper" in README.
                    results['beta'] += self.args.beta_min

                    results['transient_sigmas'] = transient_sigmas

                    static_alphas_shifted = \
                        torch.cat([torch.ones_like(static_alphas[:, :1]), 1-static_alphas], -1)
                    static_transmittance = torch.cumprod(static_alphas_shifted[:, :-1], -1)
                    static_only_weights = static_alphas * static_transmittance
                    static_only_weights_sum = static_only_weights.sum(dim=1, keepdim=True) 
                    results[f'static_only_weights'] = static_only_weights    
                    results[f'static_only_acc_map'] = static_only_weights_sum    

                    transient_rgb_map = (transient_weights[:,:,None]*transient_rgbs).sum(dim=1) # N_rays, 3
                    results['transient_rgb_map'] = transient_rgb_map

            else:
                results['beta'] = (weights*static_betas).sum(dim=1) # N_rays
                # Add beta_min AFTER the beta composition. Different from eq 10~12 in the paper.
                # See "Notes on differences with the paper" in README.
                results['beta'] += self.args.beta_min


            normed_weights = weights / (weights_sum[:,None]+1e-9)
            results['depth_map'] = (normed_weights*z_vals)[:,:-1].sum(1) * (weights_sum>0.01) # N_rays
            depth_diff = torch.pow((z_vals-results['depth_map'][:,None]), 2)
            unit_depth = (z_vals.max(dim=-1)[0]-z_vals.min(dim=-1)[0]) 
            results['depth_var_map'] = (normed_weights*depth_diff)[:,:-1].sum(dim=1) / ((unit_depth * unit_depth)/5000+1e-9) # N_rays

            results['is_edge'] = (weights_sum>0.1)*torch.maximum((weights_sum<0.90), (results['depth_var_map']>1))
        elif raw.shape[1] != 1: ### multiple sample points per ray. normal mode         
            # all weights and transient color are inherited from the geometry model
            weights = geom_cache['weights'] # N_rays, N_samples_
            weights_sum = geom_cache['acc_map'] # N_rays
            if output_transient:
                transient_sigmas = geom_cache['transient_sigmas'] # N_rays, N_samples_
                static_weights = geom_cache['static_weights'] # N_rays, N_samples_
                static_only_weights = geom_cache['static_only_weights'] # N_rays, N_samples_
                static_only_weights_sum = geom_cache['static_only_acc_map'] # N_rays, 1

                transient_weights = 1-torch.exp(-transient_sigmas) # N_rays, N_samples_
                results[f'transient_acc_map'] = (transient_weights*static_weights).sum(dim=-1)  # N_rays
                transient_rgbs = raw[..., color_dim+1:color_dim+4] # N_rays, N_samples_, 3
                transient_betas = raw[..., color_dim+5] # N_rays, N_samples_      

                if not self.args.transient_lerp_mode:
                    transient_rgb_map = geom_cache['transient_rgb_map'] 
            else:
                static_only_weights = weights # N_rays, N_samples_
                static_only_weights_sum = weights_sum[...,None] # N_rays, 1
                geom_cache['static_only_acc_map'] = static_only_weights_sum # N_rays, 1
        else: ### one sample point per ray. expected depth mode
            # in the expected depth mode, all weights and transient color are inherited from the geometry model
            weights = geom_cache['acc_map'][:,None] # N_rays, 1
            weights_sum = geom_cache['acc_map'] # N_rays
            if output_transient:
                transient_sigmas = raw[..., color_dim+4] 
                transient_rgbs = raw[..., color_dim+1:color_dim+4]   
                transient_betas = raw[..., color_dim+5]      

                static_weights = geom_cache['static_acc_map'][:,None]
                static_only_weights = geom_cache['static_only_acc_map']
                static_only_weights_sum = geom_cache['static_only_acc_map']

                if self.args.transient_lerp_mode:
                    results['transient_sigmas'] = transient_sigmas   
                    transient_weights = 1-torch.exp(-transient_sigmas)
                    results[f'transient_acc_map'] = transient_weights.sum(dim=-1) 
                    results['beta'] = (transient_weights*transient_betas).sum(dim=1) # N_rays
                    # Add beta_min AFTER the beta composition. Different from eq 10~12 in the paper.
                    # See "Notes on differences with the paper" in README.
                    results['beta'] += self.args.beta_min
                else:
                    transient_rgb_map = geom_cache['transient_rgb_map'] 
            else:
                static_only_weights = weights
                static_only_weights_sum = weights_sum[...,None]
                geom_cache['static_only_acc_map'] = static_only_weights_sum
                
            results['depth_map'] = (weights*z_vals).sum(1) / (weights_sum+1e-9) # N_rays
                    
        # Calculate Static Color
        if not geom_only:
            if not rendering_mode: # implicit appearance
                static_rgbs = raw[..., :3] # N_rays, N_samples_, 3
            else: 
                N_rays, N_samples = z_vals.shape
                static_albedo = raw[..., :3].reshape(-1, 3) # N_rays*N_samples_, 3
                static_normal = raw[..., 3:6].reshape(-1, 3) # N_rays*N_samples_, 3
                static_specular = raw[..., 6:7].reshape(-1, 1) # N_rays*N_samples_, 1
                static_glossiness = raw[..., 7:8].reshape(-1, 1) + self.args.min_glossiness # N_rays*N_samples_, 1

                ### Rendering w/ Spherical Harmonics Intergration 
                env_light = env_light[:, None, :, :].repeat([1, N_samples, 1, 1]).reshape(N_rays*N_samples, -1, 3)
                
                # Diffusion Color: from the first two levels of SH only
                diffuse_rgb = sh.render_irrandiance_sh_sum(
                                    env_light[:,:9,:3], static_normal, 
                                    fast=True, pow_num=self.sh_pow_num, s=self.sh_s)
                diffuse_rgb = diffuse_rgb.reshape(static_albedo.shape).clamp(0) # N_rays*N_samples, 3
                static_rgbs = static_albedo * diffuse_rgb # N_rays*N_samples, 3
                
                # Specular Color: Use fast intergration proposed by Ramamoorthi et.at., 2001.
                if self.args.use_specular:
                    # reflection formula: w_i = 2 * |w_o * n| * n - w_o
                    rays_d = utils.normalize(rays_d[:, None, :].repeat([1, N_samples, 1]).reshape(N_rays*N_samples, 3)) # N_rays*N_samples, 3
                    cos_theta = -(rays_d*static_normal).sum(dim=-1, keepdim=True) # N_rays*N_samples, 1
                    reflect_d = utils.normalize(2 * cos_theta * static_normal + rays_d) # N_rays*N_samples, 3

                    order_coeff = torch.arange(0, env_light.shape[1], device=env_light.device)[None,:,None] # 1, N_sh, 1
                    order_coeff = torch.pow(order_coeff, 0.5).floor()
                    s = static_glossiness[:, None, :] # N_rays*N_samples, 1, 1
                    sh_coeff = torch.exp(-order_coeff * order_coeff / 2 / s) * env_light # N_rays*N_samples, N_sh, 3
                    specular_rgb = sh.fast_sh_sum(
                                        self.sh_order, self.sh_pow_num, self.sh_s, 
                                        sh_coeff, xyz=reflect_d).clamp(0) 
                    specular_rgb = (static_specular * specular_rgb).reshape(static_albedo.shape)
                    static_rgbs = static_rgbs + specular_rgb
 
                specular_rgb = specular_rgb.clamp(0, 1).reshape(N_rays, N_samples, -1) 
                static_rgbs = static_rgbs.clamp(0, 1).reshape(N_rays, N_samples, -1) 
                # Tone-Mapping
                static_rgbs = utils.safe_pow(static_rgbs, 1/gamma[:,None,None].clamp(0.1)).clamp(0, 1)
                static_albedo = static_albedo.reshape(N_rays, N_samples, -1) 
                static_normal = static_normal.reshape(N_rays, N_samples, -1) 
                static_specular = static_specular.reshape(N_rays, N_samples, -1)
                static_glossiness = static_glossiness.reshape(N_rays, N_samples, -1)
       
            # Blending with transient color, background color
            if output_transient:
                if self.args.transient_lerp_mode: # Lerp-Based Blending
                    static_rgb_map = (static_weights[:,:,None] * (1-transient_weights[...,None]) * static_rgbs).sum(dim=1) # N_rays, 3
                    transient_rgb_map = (static_weights[:,:,None] * transient_weights[...,None] * transient_rgbs).sum(dim=1) # N_rays, 3
                else: # Nerf-W's Original Blending
                    static_rgb_map = (static_weights[:,:,None]*static_rgbs).sum(dim=1) # N_rays, 3

                if self.args.white_bkgd:
                    static_rgb_map += (1-weights_sum[:, None])*bkgd

                results['rgb_map'] = static_rgb_map + transient_rgb_map

                # Debug only: paint pixels outside of the bounding box to red
                if testing and self.args.debug_green_bkgd:
                    bkgd2 = torch.from_numpy(np.array([1,0,0])).type_as(static_rgbs)[None,:]
                    results['rgb_map'] = (1-out_of_box)*results['rgb_map']+out_of_box*bkgd2

                # Compute also static and transient rgbs when only one field exists.
                # The result is different from when both fields exist, since the transimttance
                # will change.
                static_only_rgb_map = (static_only_weights[:,:,None]*static_rgbs).sum(dim=1) # N_rays, 3
                if self.args.white_bkgd:
                    static_only_rgb_map += (1-static_only_weights_sum)*bkgd
                results['static_rgb_map'] = static_only_rgb_map
            else: # no transient field
                rgb_map = (weights[:,:,None]*static_rgbs).sum(dim=1) # N_rays, 3
                #print(rgb_map.min())
                if self.args.white_bkgd:
                    rgb_map += (1-weights_sum[:, None])*bkgd
                results['rgb_map'] = rgb_map                    
                results['static_rgb_map'] = rgb_map
            
            # save intermediate results for debugging.
            if model_type == "rendering":
                # diffuse-only rgb
                diffuse_rgb_map = (static_only_weights[:,:,None]*diffuse_rgb.reshape(N_rays, N_samples, -1)).sum(dim=1) # N_rays, 3
                results['diffuse_rgb_map'] = diffuse_rgb_map
                
                static_albedo_map = (static_only_weights[:,:,None]*static_albedo).sum(dim=1) # N_rays, 3
                static_albedo_map = utils.safe_pow(static_albedo_map, 1/2.4).detach().clamp(0, 1)
                results['albedo_map'] = static_albedo_map
                
                static_normal_map = (static_only_weights[:,:,None]*static_normal).sum(dim=1) # N_rays, 3
                results['normal_map'] = utils.normalize(static_normal_map)
                results['normal_map_weighted'] = results['normal_map'] * static_only_weights_sum

                # this step is to filter out background pixels
                if self.args.use_specular:
                    static_spec_map = (static_only_weights[:,:,None]*static_specular).sum(dim=1) # N_rays, 1
                    static_spec_map = utils.safe_pow(static_spec_map, 1/2.4).detach().clamp(0, 1)
                    results['spec_map'] = static_spec_map[:,[0,0,0]]
                    static_spec_map = (static_only_weights[:,:,None]*specular_rgb).sum(dim=1) # N_rays, 1
                    static_spec_map = utils.safe_pow(static_spec_map, 1/2.4).detach().clamp(0, 1)
                    results['spec_rgb_map'] = static_spec_map[:,[0,0,0]]
                    static_glossiness_map = (static_only_weights[:,:,None]*static_glossiness).sum(dim=1) # N_rays, 3
                    results['glossiness_map'] = (1/(static_glossiness_map+1))[:,[0,0,0]]
        
        return results
    
    def _render_rays(self, ray_batch,
                    N_samples,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    raw_noise_std=0.,
                    light_param=None,
                    testing=False,
                    **kwargs):
        """Volumetric rendering.
        Args:
            ray_batch: array of shape [batch_size, ...]. All information necessary
                for sampling along a ray, including: ray origin, ray direction, min
                dist, max dist, img_id, and unit-magnitude viewing direction.
            N_samples: int. Number of different times to sample along each ray.
            lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
            perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
                random points in time.
            N_importance: int. Number of additional times to sample along each ray.
                These samples are only passed to networks['model_fine'].
            raw_noise_std: ...
            light_param: environment lighting and gamma input for relighting
            testing: whether in testing mode.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
            disp_map: [num_rays]. Disparity map. 1 / depth.
            acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
            raw: [num_rays, num_samples, 4]. Raw predictions from model.
            rgb0: See rgb_map. Output for coarse model.
            disp0: See disp_map. Output for coarse model.
            acc0: See acc_map. Output for coarse model.
            z_std: [num_rays]. Standard deviation of distances along ray for each
                sample.
            ...
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
        img_ids = ray_batch[:, 8].long() # [N_rays]
        viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 9 else None
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
        near, far = bounds[...,0], bounds[...,1] # [-1,1]
        # Decompose the inputs

        # Sample depth points
        t_vals = torch.linspace(0., 1., steps=N_samples, device=near.device)
        if not lindisp:
            z_vals = near * (1.-t_vals) + far * (t_vals)
        else:
            z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])
        
        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand_like(z_vals)

            t_rand = t_rand.to(lower.device)
            z_vals = lower + (upper - lower) * t_rand

        out_of_box = (z_vals[:, 0] > z_vals[:, 1])[:,None].float() # if near > far, then z_vals are descending
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

        if self.model_type == "rendering":
            if light_param is not None: # lighting coeffs from input
                env_light = light_param[0][None,:,:].repeat(rays_d.shape[0], 1, 1)
                gamma = light_param[1].repeat(rays_d.shape[0])
            else: # lighting coeffs from parameters
                env_light = self.env_lights[img_ids] # [N_rays, order^2, 3]
                gamma = self.gamma[img_ids]
        else:
            env_light = None
            gamma = 2.4

        raw = self._inference(self.networks['model_geom_coarse'], pts, viewdirs)
        results_coarse = self.raw2outputs("geometry", raw, rays_d, z_vals, env_light, raw_noise_std, 
                                            out_of_box=out_of_box, testing=testing) 

        if N_importance > 0:
            weights = results_coarse['weights']

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1].detach(), N_importance, det=(perturb==0.))
            # z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            z_vals_index = torch.argsort(weights[...,1:-1], descending=True)+1
            z_vals_sorted_by_weights = torch.gather(z_vals, 1, z_vals_index)[:, :N_importance].detach()
            z_vals, _ = torch.sort(torch.cat([z_samples, z_vals_sorted_by_weights], -1), -1)
            #z_vals, _ = torch.sort(z_samples, -1)
            
            # if the ray is outside of the box, invalid all sample points
            z_vals *= (1-out_of_box) 
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            run_fn = self.networks['model_geom_coarse'] if self.networks['model_geom_fine'] is None else self.networks['model_geom_fine']
            
            a_embedded, t_embedded = None, None         
            if self.args.encode_appearance:
                if light_param is not None and self.model_type=="geometry":
                    a_embedded = light_param[0][None,:].repeat(rays_d.shape[0], 1)
                else:
                    a_embedded = self.networks['embed_a'](img_ids)
            if self.args.encode_transient:
                t_embedded = self.networks['embed_t'](img_ids)

            raw_fine = self._inference(run_fn, pts, viewdirs, a_embedded, t_embedded)
            results_geom = self.raw2outputs(self.model_type, raw_fine, rays_d, z_vals, env_light, raw_noise_std, 
                                out_of_box=out_of_box, output_transient=self.args.encode_transient, 
                                testing=testing, geom_only = self.args.use_expected_depth, gamma=gamma)

            # calculate raw outputs with difference (output from neightbor points)
            def calc_raw_with_diff(z_vals, idxs, with_diff=False, raw=None):
                assert(len(z_vals) == len(idxs))
                N_rays_idx = len(idxs)
                pts = rays_o[idxs,None,:] + rays_d[idxs,None,:] * z_vals[...,:,None] # [N_rays_idx, ?, 3]       
                results_geom_all = {}       

                if with_diff:
                    z_vals_all = torch.cat([z_vals, z_vals])
                    pts_diff = (pts + torch.rand_like(pts)*0.01).detach()
                    pts_all = torch.cat([pts, pts_diff])
                    # duplicate all variables to handle smoothing loss
                    for k in results_geom:
                        results_geom_all[k] = torch.cat([results_geom[k][idxs], results_geom[k][idxs]]).detach()
                    viewdirs_all = torch.cat([viewdirs[idxs], viewdirs[idxs]])
                    a_embedded_all = torch.cat([a_embedded[idxs], a_embedded[idxs]]) if a_embedded is not None else None
                    t_embedded_all = torch.cat([t_embedded[idxs], t_embedded[idxs]]) if t_embedded is not None else None
                    rays_d_all = torch.cat([rays_d[idxs], rays_d[idxs]])
                    out_of_box_all = torch.cat([out_of_box[idxs], out_of_box[idxs]])
                    if env_light is None:
                        env_light_all = None
                    else:
                        gamma_all = torch.cat([gamma[idxs], gamma[idxs]])
                        env_light_all = torch.cat([env_light[idxs], env_light[idxs]])
                else:
                    z_vals_all = z_vals
                    pts_all = pts
                    # duplicate all variables to handle smoothing loss
                    for k in results_geom:
                        results_geom_all[k] = results_geom[k][idxs].detach()
                    viewdirs_all = viewdirs[idxs]
                    a_embedded_all = a_embedded[idxs] if a_embedded is not None else None
                    t_embedded_all = t_embedded[idxs] if t_embedded is not None else None
                    rays_d_all = rays_d[idxs]
                    out_of_box_all = out_of_box[idxs]
                    if env_light is None:
                        env_light_all = None
                    else:
                        gamma_all = gamma[idxs]
                        env_light_all = env_light[idxs]
                
                if raw is None:   
                    raw = self._inference(run_fn, pts_all, viewdirs_all, 
                                a_embedded_all, t_embedded_all)
                results_final = self.raw2outputs(
                    self.model_type, raw, rays_d_all, z_vals_all, env_light_all, raw_noise_std, 
                    out_of_box=out_of_box_all,output_transient=self.args.encode_transient, 
                    testing=testing, geom_only=False, geom_cache=results_geom_all, gamma=gamma_all)
                
                results = {}
                for k in results_final:
                    results[k] = results_final[k][:N_rays_idx]
                    results[k+'_diff'] = results_final[k][N_rays_idx:] if with_diff else results[k]

                return results

            if not self.args.use_expected_depth:
                results = results_geom
            else:
                # whether the rays are at the edges of the object
                is_edge_idx = torch.nonzero(results_geom['is_edge'])[...,0].detach()
                is_inside_idx = torch.nonzero(~results_geom['is_edge'])[...,0].detach()

                # rays inside the object blocks
                if len(is_inside_idx) > 0:
                    z_vals_inside = results_geom['depth_map'][is_inside_idx, None].detach()
                    # filter out invalid pixels
                    z_vals_inside *= (1-out_of_box[is_inside_idx])
                    z_vals_inside *= (results_geom['acc_map'][is_inside_idx, None].detach() > 0.1) 
                    results_inside = calc_raw_with_diff(
                        z_vals_inside, is_inside_idx, with_diff=True)

                # rays at the edges
                if is_edge_idx.shape[0] > 0: 
                    z_vals_edge = z_vals[is_edge_idx]
                    z_vals_edge *= (1-out_of_box[is_edge_idx])
                    results_edge = calc_raw_with_diff(
                        z_vals_edge, is_edge_idx, with_diff=False, raw=raw_fine[is_edge_idx])   
                    idx_perm_temp = torch.cat([is_edge_idx, is_inside_idx])
                    idx_perm = torch.arange(len(idx_perm_temp)).type_as(idx_perm_temp)
                    idx_perm[idx_perm_temp.long()] = idx_perm.clone()
                    idx_perm = idx_perm.detach()     

                # blend results together
                if min(len(is_inside_idx), len(is_edge_idx)) > 0: # need to blend
                    results = {}
                    for k in results_inside:
                        if k=='weights' or k not in results_edge:
                            continue 
                        elif 'transient_sigmas' in k:
                            results[k] = torch.cat([results_edge[k], results_inside[k].repeat(1, results_edge[k].shape[1])], dim=0)[idx_perm]
                        else:
                            results[k] = torch.cat([results_edge[k], results_inside[k]], dim=0)[idx_perm]
                elif len(is_edge_idx) > 0: 
                    results = results_edge
                else:
                    results = results_inside

        if testing:
            for k in results:
                results[k] = results[k].detach()

        if N_importance > 0:
            for k, v in results_coarse.items():
                results[k+"_coarse"] = v if self.model_type == "geometry" else v.detach() 
            results['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in results:
            if (torch.isnan(results[k]).any() or torch.isinf(results[k]).any()):
                print(f"! [Numerical Error] {k} contains nan or inf.")

        return results
        
    def _render_sigma(self, pts):
        """Calculate density at different 3D positions.
        Args:
            pts: [num_points, 3]. 3D positions.
        Returns:
            sigma: [num_points, 3]. static density at positions
        """

        N_pts = pts.shape[0]
        viewdirs = pts.clone() # N_pts, 3

        if self.args.encode_appearance:
            a_embedded = self.networks['embed_a'](torch.zeros_like(pts[...,0].long()))
        else:
            a_embedded = None
        output_transient = self.args.encode_transient
        if output_transient:
            t_embedded = self.networks['embed_t'](torch.zeros_like(pts[...,0].long()))
        else:
            t_embedded = None

        run_fn = self.networks['model_geom_coarse'] if self.networks['model_geom_fine'] is None else self.networks['model_geom_fine']
        raw = self._inference(run_fn, pts[:,None,:], viewdirs, a_embedded, t_embedded, freeze_geometry=True)
        return raw[...,3]

    def calculate_loss(self, result_dict, gt_color, gt_normal, rays_is_bg, bds_dict, img_id):
        ret_dict = {}

        ### Common Losses:

        # Color Loss L_c: 
        if 'rgb_map_coarse' in result_dict:
            #print(extras['rgb_map_coarse'].min())
            ret_dict['c_l'] = 0.5 * ( ((result_dict['rgb_map_coarse']-gt_color)**2) ).mean() # utils.img2mse(extras['rgb_map_coarse'], gt_color)

        if 'beta' not in result_dict: # no transient head, normal MSE loss
            ret_dict['f_l'] = 0.5 * utils.img2mse(result_dict['rgb_map'], gt_color)
        else:
            ret_dict['f_l'] = \
                ((result_dict['rgb_map']-gt_color)**2/(2*result_dict['beta'].unsqueeze(1)**2)).mean()

            ret_dict['b_l'] = 3 + torch.log(result_dict['beta']).mean() # +3 to make it positive
        
        # Transient Regularity Loss L_tr: 
        if 'transient_sigmas' in result_dict:
            ret_dict['tr_l'] = self.args.lambda_tr * result_dict['transient_sigmas'].mean()
            
        ### Geometry Network's Losses:

        if self.model_type == "geometry":
            # Sihlouette Loss L_sil: 
            if self.args.lambda_sil > 0:
                ret_dict['sil_l'] = self.args.lambda_sil * F.binary_cross_entropy(result_dict['acc_map']*0.98+0.01, (1-rays_is_bg[:,0]))
                ret_dict['sil_l'] += self.args.lambda_sil * F.binary_cross_entropy(result_dict['acc_map_coarse']*0.98+0.01, (1-rays_is_bg[:,0]))            
            
            # Camera Regularity Loss L_cam:
            if self.args.optimize_camera:
                ret_dict['cam_l'] = self.args.lambda_cam * (self.cam_df.norm(dim=-1).mean()+self.cam_dR.norm(dim=-1).mean()+self.cam_dt.norm(dim=-1).mean())
            
        ### Rendering Network's Losses

        if self.model_type == "rendering":
            # Normal Loss L_n:
            if 'normal_map' in result_dict:
                gt_normal_norm = gt_normal.norm(dim=-1, keepdim=True)
                ret_dict['n_l'] = self.args.lambda_n * ((result_dict['normal_map']*gt_normal_norm-gt_normal)**2 * \
                    (1-rays_is_bg[:,0:1]) * (1-result_dict['is_edge'][:,None].float())).mean()

            # Normal Smoothness Loss L_sm:
            if self.args.use_expected_depth:
                weights = (result_dict['static_only_acc_map'] * result_dict['static_only_acc_map_diff']).detach()
                ret_dict['smooth_n_l'] = self.args.lambda_smooth_n * F.l1_loss(result_dict['normal_map'] * weights, result_dict['normal_map_diff'] * weights)

            # Other Regularity Losses L_reg (Specularity, SH coeffs, Gamma):
            if 'spec_map' in result_dict:
                ret_dict['spec_l'] = self.args.lambda_spec * utils.img2mse(result_dict['spec_map'], 0)
            if self.args.lambda_light > 0:
                sample_angle = torch.rand_like(result_dict['rgb_map'][...,:2])
                sample_angle[...,0] = sample_angle[...,0] * 2 * pi
                sample_angle[...,1] = torch.acos(1 - 2 * sample_angle[...,1])
                sample_xyz = sh.angle2xyz(sample_angle)
                sampled_light = sh.fast_sh_sum(
                    self.sh_order, self.sh_pow_num, self.sh_s, 
                    self.env_lights[img_id[...,0].long()], xyz=sample_xyz)
                ret_dict['light_l'] = self.args.lambda_light * (torch.relu(-sampled_light-0.01)**2).mean()
                ret_dict['gamma_l'] = self.args.lambda_light * ((self.gamma[img_id[...,0].long()]-2.4)**2).mean()
            
        loss = 0
        for v in ret_dict.values():
            loss += v
        ret_dict['loss'] = loss

        with torch.no_grad():
            recon_loss = utils.img2mse(result_dict['rgb_map'], gt_color)
            ret_dict['psnr'] = utils.mse2psnr(recon_loss)

        prog_list = ['cam_l', 'f_l', 'c_l', 
            'n_l' if self.args.lambda_sil==0 else 'sil_l', 'psnr']

        return ret_dict, prog_list