
import os, sys
import shutil
import numpy as np
import imageio
import cv2
import json
import random
import time
from numpy.lib import utils
import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import math
from tqdm import tqdm, trange
import pytorch3d.transforms as transforms
import sh_functions as sh

test_number = [1,2,3]

if __name__ == "__main__":
    # Loading
    pow_num, s = sh.pre_calc_sh_mat()
    if os.path.exists("./assets/sh_test"):
        shutil.rmtree("./assets/sh_test")
    os.makedirs("./assets/sh_test", exist_ok=True)
    env_order = 30
    fast_order = 3
    img_w = 400
    img_h = 200
    img = imageio.imread("./assets/sh_test_env.jpeg")
    img = torch.from_numpy(cv2.resize(img, (img_w, img_h))).float() / 255.0

    img = img.cuda()
    pow_num = pow_num.cuda()
    s = s.cuda()

    num_lm_env = (env_order+1)*(env_order+1)
    num_lm_fast = (fast_order+1)*(fast_order+1)
    coeffs = sh.project_environment(env_order, img)
    fast_coeffs = coeffs[:num_lm_fast]

    # Test 1: Use Spherical Harmonics to reconstruct image
    # This test covers two operations: sh_eval and sh_eval_sum
    if 1 in test_number:
        print("Test 1: Reconstruct env map")
        img_recon = sh.unproject_environment(env_order, coeffs, img.shape[0], img.shape[1])
        img_recon = (img_recon.clamp(0, 1)*255.0).cpu().numpy().astype(np.uint8)
        imageio.imwrite("./assets/sh_test/1_recon_full.jpeg", img_recon)  

        img_recon_low = sh.unproject_environment(fast_order, fast_coeffs, img.shape[0], img.shape[1])
        img_recon_low = (img_recon_low.clamp(0, 1)*255.0).cpu().numpy().astype(np.uint8)
        imageio.imwrite("./assets/sh_test/1_recon_loworder.jpeg", img_recon_low)  

        img_recon_fast = sh.unproject_environment(fast_order, fast_coeffs, img.shape[0], img.shape[1], 
                                                    fast=True, pow_num=pow_num, s=s)
        img_recon_fast = (img_recon_fast.clamp(0, 1)*255.0).cpu().numpy().astype(np.uint8)
        imageio.imwrite("./assets/sh_test/1_recon_fast.jpeg", img_recon_fast)     

        img_diff = np.abs(img_recon_low*1.0 - img_recon_fast*1.0)
        imageio.imwrite("./assets/sh_test/1_diff.jpeg", img_diff)  
        assert(img_diff.mean() < 1)
        print("Test 1 Passed")

    # Test 2: Use SH function to calculate irradiance from environment map.
    if 2 in test_number:
        print("Test 2: Calculate irradiance")
        rotate_irradiance = sh.render_irrandiance_map_rotate(fast_coeffs, 100, 200)
        rotate_irradiance = (rotate_irradiance.clamp(0, 1)*255.0).cpu().numpy().astype(np.uint8)
        fast_irradiance = sh.render_irrandiance_map_sh_sum(fast_coeffs, 100, 200, True, pow_num, s)
        fast_irradiance = (fast_irradiance.clamp(0, 1)*255.0).cpu().numpy().astype(np.uint8)
        gt_irradiance = sh.render_irrandiance_map_direct(img, 100, 200)
        gt_irradiance = (gt_irradiance.clamp(0, 1)*255.0).cpu().numpy().astype(np.uint8)
        imageio.imwrite("./assets/sh_test/2_irradiance_gt.jpeg", gt_irradiance)     
        imageio.imwrite("./assets/sh_test/2_irradiance_rotate.jpeg", rotate_irradiance)     
        imageio.imwrite("./assets/sh_test/2_irradiance_fast.jpeg", fast_irradiance)   
        img_rotate_diff = np.abs(gt_irradiance*1.0 - rotate_irradiance*1.0).astype(np.uint8)
        img_fast_diff = np.abs(gt_irradiance*1.0 - fast_irradiance*1.0).astype(np.uint8)
        imageio.imwrite("./assets/sh_test/2_irradiance_rotate_diff.jpeg", img_rotate_diff)  
        imageio.imwrite("./assets/sh_test/2_irradiance_fast_diff.jpeg", img_fast_diff)  
        assert(img_rotate_diff.mean() < 1)
        assert(img_fast_diff.mean() < 1)
        print("Test 2 Passed")

    # Test 3: Use SH function to render a ball with phong specular brdf.
    if 3 in test_number: 
        print("Test 3: Calculate Phong BRDF")   
        # irr_img = sh.render_phong_irrandiance_map_alternative(fast_order, fast_coeffs, 
        #                                                         500, 500, True, pow_num, s)
        sh_phong_img = sh._render_phong_map_sh_sum(env_order, coeffs, 400, 400)
        gt_phong_img = sh._compute_phong_direct(img, 400, 400, s=32)
        #irr_img = compute_explicit_diffuse_irradiance(img, 200, 400)
        #irr_img = render_diffuse_irrandiance_map_alternative(coeffs, 960, 1920)
        sh_render_img = sh._render_ball(sh_phong_img, 500, is_reflect=True)
        sh_render_img = (sh_render_img.clamp(0, 1)*255.0).cpu().numpy().astype(np.uint8)
        gt_render_img = sh._render_ball(gt_phong_img, 500, is_reflect=True)
        gt_render_img = (gt_render_img.clamp(0, 1)*255.0).cpu().numpy().astype(np.uint8)
        img_phong_diff = np.abs(sh_render_img*1.0 - gt_render_img*1.0).astype(np.uint8)
        imageio.imwrite("./assets/sh_test/3_phong_render_fast.jpeg", sh_render_img)    
        imageio.imwrite("./assets/sh_test/3_phong_render_gt.jpeg", gt_render_img)    
        imageio.imwrite("./assets/sh_test/3_phong_render_diff.jpeg", img_phong_diff)   
        assert(img_phong_diff.mean() < 1)
        print("Test 3 Passed")
