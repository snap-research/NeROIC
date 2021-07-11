
def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

# -- Common options
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory'),
    parser.add_argument("--num_epochs", type=int, default=200, 
                        help='train how many epoches'),
    parser.add_argument("--num_gpus", type=int, default=1, 
                        help='use how many gpus')
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--debug_green_bkgd", action='store_true', 
                        help='set to render synthetic data on a green bkgd (need to set white bkgd true first)')
    
# -- Network options    
    parser.add_argument("--model", type=str, choices=['NeROIC'], default='NeROIC', 
                        help='name of the model')
    parser.add_argument("--model_type", type=str, choices=["geometry", "rendering"], required=True,
                        help='Stage(Type) of the model')
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')

# -- Rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    
# -- NeRF-W options
    parser.add_argument("--encode_appearance", action='store_true', 
                        help='train nerf-w with appearance encoding')
    parser.add_argument("--encode_transient", action='store_true', 
                        help='train nerf-w with transient encoding')
    parser.add_argument('--N_vocab', type=int, default=1000,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument('--N_a', type=int, default=48,
                        help='number of embeddings for appearance')
    parser.add_argument('--N_tau', type=int, default=16,
                        help='number of embeddings for transient objects')
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='minimum color variance for each ray')

# -- Geometry network options
    parser.add_argument('--optimize_camera', action='store_true',
                        help='optimize camera at the same time')

# -- Normal extraction layer options
    parser.add_argument("--normal_smooth_alpha", type=float, default=1., help='smoothing parameter for normal extraction')

# -- Rendering network options
    parser.add_argument("--use_expected_depth", action='store_true', 
                        help='if specified, use expected depth instead of all sample points for sh model')
    parser.add_argument("--min_glossiness", type=float, default=1., help='minimum glossiness of BRDF')
    parser.add_argument("--use_specular", action='store_true', 
                        help='use specular shading in rendering')
    parser.add_argument('--transient_lerp_mode', action='store_true',
                        help='use lerp to blend transient rgb and static rgb')

# -- Training options
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--scheduler", type=str, choices=["cosine", "multistep"], default="cosine", help='type of scheduler')
    parser.add_argument("--decay_epoch", type=int, nargs="+", default=10, 
                        help='epochs that needed for decaying')
    parser.add_argument("--decay_gamma", type=float, default=0.1, 
                        help='gamma value of lr decaying')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*32, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--load_prior", action='store_true', 
                        help='is specified, load model from ft_path as a prior, then train from scratch')

# -- Testing options
    parser.add_argument('--test_img_id', type=int, default=0,
                        help='the id (of transient and lighting) used for testing image')
    parser.add_argument('--test_split', type=str,  choices=["val", "testtrain"], default="testtrain",
                        help='which split of poses is tested')
    parser.add_argument('--test_optimize_steps', type=int, default=0,
                        help='Steps for lighting/camera optimization during testing')
    parser.add_argument("--test_env_filename", type=str, default='', 
                        help='path of the testing environment map'),
    parser.add_argument("--test_env_maxthres", type=float, default=20, 
                        help='maximum radiance of the env map'),

# -- Loss options
    ## common
    parser.add_argument("--lambda_sil", type=float,
                        default=0, help='weight of silhouette loss') 
    parser.add_argument('--lambda_tr', type=float, default=0.01,
                        help='weight of transient regularity loss')
    ## geometry
    parser.add_argument('--lambda_cam', type=float, default=0.01,
                        help='weight of camera loss')

    ## rendering
    parser.add_argument("--lambda_n", type=float,
                        default=0, help='weight of normal loss') 
    parser.add_argument("--lambda_smooth_n", type=float,
                        default=0.5, help='weight of normal smoothiness loss') 

    parser.add_argument("--lambda_spec", type=float,
                        default=0, help='weight of specular regularity loss') 
    parser.add_argument("--lambda_light", type=float,
                        default=5, help='weight of light positive regularizaion loss') 

# -- Dataset options
    parser.add_argument("--dataset_type", type=str, choices=['llff', 'nerd_real'],  default='llff', 
                        help='options: llff, nerd_real')
    parser.add_argument("--test_intv", type=int, default=8, 
                        help='will take every 1/N images as test set.')
    parser.add_argument("--test_offset", type=int, default=1, 
                        help='index of the first test image')
    parser.add_argument("--train_limit", type=int, default=-1, 
                        help='the limitation of training images')
    parser.add_argument("--multiple_far", type=float, default=1.2, help='multiple of far distance')
    parser.add_argument("--have_mask", action='store_true', 
                        help='if the dataset contains mask')
    parser.add_argument("--mask_ratio", type=float, default=0, 
                        help='ratio between foreground/background rays (fg:bg = 1:N)')
    parser.add_argument("--rays_path", type=str, default="", 
                        help='cached rays file(with normal, etc.)')
    parser.add_argument("--test_resolution", type=int, default=-1, 
                        help='resolution of the testing images. If set to -1, use the maximum resolution of training images')

# -- LLFF flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--width", type=int, default=0, 
                        help='downsample width for LLFF images. Dafault set to 0 (no downsampling)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--use_bbox", action='store_true', 
                        help='use bounding box of the point cloud from SfM')

# -- Logging/Saving options
    parser.add_argument("--i_testset", type=int, default=-1, 
                        help='frequency of testset saving. -1 means test on the end of every epoch')
    parser.add_argument("--i_testepoch", type=int, default=1, 
                        help='frequency of testset saving related to epoch')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of test video saving')
    parser.add_argument("--i_traintest", type=int, default=50000, 
                        help='frequency of testing one train poses')
    parser.add_argument("--N_test_pose", type=int, default=12, 
                        help='number of test poses')
                        
    parser.add_argument("--verbose", action='store_true', help='output additional infos')
    return parser
