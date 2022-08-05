import numpy as np
import os
import sys
import imageio
# import skimage.transform
import pickle
import sys
import argparse

import colmap_read_model as read_model
from colmap_wrapper import run_colmap

def load_colmap_data(realdir):
    
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    # cam = camdata[camdata.keys()[0]]

    cam_hwf = {}
    for cam_k, cam in camdata.items():
        # w, h, f = factor * w, factor * h, factor * f
        h, w, f = cam.height, cam.width, cam.params[0]
        print( 'Cameras', h, w, f)
        cam_hwf[cam_k] = np.array([h,w,f]).reshape([3,1])
    
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    w2c_mats = []
    bottom = np.array([0,0,0,1.]).reshape([1,4])

    names = [imdata[k].name for k in imdata]
    names = [s.replace("images_unmasked", "images") for s in names]
    print( 'Images #', len(names))
    perm = np.argsort(names)
    im_keys = []
    im_names = []
    hwf = []
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
        im_keys.append(k)
        hwf.append(cam_hwf[im.camera_id])
        # im_names.append(im.)

    hwf = np.stack(hwf, axis=-1)
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    
    poses = c2w_mats[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, hwf], 1)
    
    points3dfile = os.path.join(realdir, 'sparse/0/points3D.bin')
    pts3d = read_model.read_points3d_binary(points3dfile)   

    for k in pts3d:
        for i, ind in enumerate(pts3d[k].image_ids):
            if ind not in im_keys:
                print('ERROR: the correct camera poses for current points cannot be accessed')
                return
            pts3d[k].image_ids[i] = im_keys.index(ind)+1

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    
    return poses, pts3d, perm, names


def save_poses(basedir, poses, pts3d, perm, names):
    pts_arr = []
    vis_arr = []
    name_valid = ["video" not in x for x in names]
    valid_perm = [x for x in perm if name_valid[x]]
    for k in pts3d:
        valid_pt = 0
        cams = [0] * poses.shape[-1]
        for ind in pts3d[k].image_ids:
            ind = ind-1
            if ind < len(cams) and name_valid[ind]:
                cams[ind] = 1
                valid_pt = 1
        if valid_pt:  
            pts_arr.append(pts3d[k].xyz)
            vis_arr.append(cams)

    pts_arr = np.array(pts_arr)
    vis_arr = np.array(vis_arr)
    print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )
    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)
    valid_z = zvals[vis_arr==1]
    print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean() )
    
    save_arr = []
    save_name = []
    for i in valid_perm:
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis==1]
        #print(zs)
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        # print( i, close_depth, inf_depth )
        
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
        save_name.append(names[i])

    save_arr = np.array(save_arr)
    
    pickle.dump(pts_arr, open(os.path.join(basedir, 'pts.pkl'), "wb"))

    #np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)
    np.save(os.path.join(basedir, 'poses_bounds.npy'), save_arr)

    # pickle.dump(save_name, open(os.path.join(basedir, 'image_names.pkl'), "wb"))
    save_name = save_name[0:1]+["\n"+x for x in save_name[1:]]
    open(os.path.join(basedir, 'image_names.txt'), "w").writelines(save_name)
     
def gen_poses(basedir, match_type):
    
    files_needed = ['{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']]
    if os.path.exists(os.path.join(basedir, 'sparse/0')):
        files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
    else:
        files_had = []
    if not all([f in files_had for f in files_needed]):
        print( 'Need to run COLMAP' )
        run_colmap(basedir, match_type)
    else:
        print('Don\'t need to run COLMAP')
        
    print( 'Post-colmap')

    poses, pts3d, perm, names = load_colmap_data(basedir)
    save_poses(basedir, poses, pts3d, perm, names)
    
    return True
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--match_type', type=str, 
                        default='exhaustive_matcher', help='type of matcher used.  Valid options: \
                        exhaustive_matcher sequential_matcher.  Other matchers not supported at this time')
    parser.add_argument('--scenedir', type=str,
                        help='input scene directory')
    args = parser.parse_args()

    if args.match_type != 'exhaustive_matcher' and args.match_type != 'sequential_matcher':
        print('ERROR: matcher type ' + args.match_type + ' is not valid.  Aborting')
        sys.exit()
    gen_poses(args.scenedir, args.match_type)