import numpy as np
import os
import sys
import imageio
# import skimage.transform
import pickle
import sys
import argparse
import cv2
import glob

import colmap_read_model as read_model
from colmap_wrapper import run_colmap
    
def is_image(f):
    f = f.lower()
    return f.endswith("jpg") or f.endswith("jpeg") or f.endswith("png")
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_type', type=str, 
                        default='exhaustive_matcher', help='type of matcher used.  Valid options: \
                        exhaustive_matcher sequential_matcher.  Other matchers not supported at this time')
    parser.add_argument('--scenedir', type=str,
                        help='input scene directory')
    args = parser.parse_args()

    image_input_dir = os.path.join(args.scenedir, "images_unmasked")
    image_output_dir = os.path.join(args.scenedir, "images")
    mask_dir = os.path.join(args.scenedir, "images_mask")
    os.makedirs(image_output_dir, exist_ok=True)

    image_list = glob.glob(os.path.join(image_input_dir, "*"))
    image_list = [x for x in image_list if is_image(x)]
    mask_list = [x.replace("images_unmasked", "images_mask") for x in image_list]

    for img_file, mask_file in zip(image_list, mask_list):
        img_name = os.path.basename(img_file)
        assert(os.path.exists(mask_file))
        img = cv2.imread(img_file)[...,:3] / 255.0
        mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), cv2.INTER_LINEAR)
        if mask.ndim == 2: # B/W:
            mask = mask[...,np.newaxis]
        mask = mask[...,0:1] / 255.0
        masked_img = img * mask + (1-mask)
        masked_img = (masked_img * 255.0).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(image_output_dir, img_name), masked_img)

    print("finished")