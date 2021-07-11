'''
    Library of Spherical Harmonics functions.
    this library is based on https://github.com/google/spherical-harmonics.
'''

import os, sys
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



### --- Helper functions

def normalize(tensor):
    return tensor / (tensor.norm(dim=-1, keepdim=True)+1e-9)

get_coef_count = lambda order: (order+1)*(order+1)
get_index = lambda l, m: l*(l+1)+m

fmod = lambda x, m: x - (m * (x/m).floor())
nearbymargin = lambda x, y: abs(x-y) < 1e-8

# factorial: x!
factorial_cache = [1, 1, 2, 6, 24, 120, 720, 5040,
                    40320, 362880, 3628800, 39916800,
                    479001600, 6227020800,
                    87178291200, 1307674368000]
factorial = lambda x: factorial_cache[x] if x <= 15 else math.factorial(x)

# double factorial: x!!
dbl_factorial_cache = [1, 1, 2, 3, 8, 15, 48, 105,
                        384, 945, 3840, 10395, 46080,
                        135135, 645120, 2027025]
double_factorial = lambda x: dbl_factorial_cache[x] if x <= 15 else torch.prod(torch.arange(x, 0, -2))



### --- Functions for SH calculation/integration

# Generate matrics for fast calculation
def pre_calc_sh_mat():
    ar = torch.arange(4)
    mat_x, mat_y, mat_z = torch.meshgrid(ar, ar, ar) # [4,4,4]
    mat = torch.stack([mat_x, mat_y, mat_z], dim=-1) # [4,4,4,3]

    # Assigning constants of the real SH polynomials
    s = torch.zeros([16, 4, 4, 4])
    # Level 0
    s[0][0][0][0] = 0.282095    # 0.282095

    # Level 1
    s[1][0][1][0] = -0.488603   # -0.488603 * y
    s[2][0][0][1] = 0.488603   # 0.488603 * z 
    s[3][1][0][0] = -0.488603   # -0.488603 * x 

    # Level 2
    s[4][1][1][0] = 1.092548   # 1.092548 * x * y
    s[5][0][1][1] = -1.092548; # -1.092548 * y * z
    s[6][2][0][0] = -0.315392; s[6][0][2][0] = -0.315392; s[6][0][0][2] = 0.315392 * 2.0   # 0.315392 * (- x * x - y * y + 2.0 * z * z)
    s[7][1][0][1] = -1.092548   # -1.092548 * x * z
    s[8][2][0][0] = 0.546274; s[8][0][2][0] = -0.546274  # 0.546274 * (x * x - y * y)

    # Level 3
    s[9][2][1][0] = -0.590044 * 3.0; s[9][0][3][0] = 0.590044 # -0.590044 * y * (3.0 * x * x - y * y)
    s[10][1][1][1] = 2.890611    # 2.890611 * x * y * z
    s[11][0][1][2] = -0.457046 * 4.0; s[11][2][1][0] = 0.457046; s[11][0][3][0] = 0.457046 # -0.457046 * y * (4.0 * z * z - x * x - y * y)
    s[12][0][0][3] = 0.373176 * 2.0; s[12][2][0][1] = -0.373176 * 3.0; s[12][0][2][1] = -0.373176 * 3.0 # 0.373176 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)
    s[13][1][0][2] = -0.457046 * 4.0; s[13][3][0][0] = 0.457046; s[13][1][2][0] = 0.457046 # -0.457046 * x * (4.0 * z * z - x * x - y * y)
    s[14][2][0][1] = 1.445306; s[14][0][2][1] = -1.445306 # 1.445306 * z * (x * x - y * y)
    s[15][3][0][0] = -0.590044; s[15][1][2][0] = 0.590044 * 3.0 # -0.590044 * x * (x * x - 3.0 * y * y)

    # select terms that are used in at least one polynomial
    valid = torch.nonzero((torch.max(s**2, dim=0)[0]>0))
    idx = torch.zeros_like(s[0]).long() # 4 x 4 x 4
    idx[valid[:,0], valid[:,1], valid[:,2]] = torch.arange(len(valid), device=valid.device) # 4 x 4 x 4
    sh_s = s[:, valid[:,0], valid[:,1], valid[:,2]] # 16 x N_valid
    sh_power_num = valid # N_valid, 3
    return sh_power_num, sh_s

def fast_sh_sum(order, power_num, s, coeffs, angle=None, xyz=None):
    '''
        power_num: [N_valid x 3], pre_computed buffer
        s: [lm x N_valid], pre_computed buffer
        
        coeffs: [B x lm x 3], input SH coefficients
        xyz: [B x 3], input coordinate
    '''
    assert(order <= 3)    
    
    if xyz is None:
        xyz = angle2xyz(angle)

    if len(xyz.shape) == 1:
        xyz = xyz[None,:]
    if coeffs.ndim == 2:
        coeffs = coeffs[None,:,:]

    coeff_count = get_coef_count(order)

    xyz_term = torch.pow(xyz[:,None,:], power_num[None,:,:]).prod(dim=-1) # B x N_Valid
    sh_val = torch.matmul(xyz_term, s.T)[:, :coeff_count] # B x lm

    return (sh_val[:,:,None] * coeffs).sum(dim=1) # B x 3

def angle2xyz(angle: torch.FloatTensor):
    # Y up
    # angle: [phi, theta], tensor shape: n, 2
    r = torch.sin(angle[..., 1])
    return torch.stack([r * torch.cos(angle[..., 0]), torch.cos(angle[..., 1]), r * torch.sin(angle[..., 0])], -1).type_as(angle)

def xyz2angle(xyz: torch.FloatTensor):
    # Y up
    return torch.stack([torch.atan2(xyz[:, 2], xyz[:, 0]), torch.acos(torch.clamp(xyz[:, 1], -1.0, 1.0))], dim=-1).type_as(xyz)

def imcoord2angle(coord: torch.FloatTensor, w, h):
    '''
        Convention between panorama coordinate and spherical coordinate: 
        Width->phi: [0 - W] -> [0 - 2pi]
        Height->theta: [0 - H] -> [0 - pi]
    '''
    return torch.stack([2.0 * math.pi * (coord[..., 0] + 0.5) / w, math.pi * (coord[..., 1] + 0.5) / h], dim=-1).type_as(coord)

def angle2imcoord(angle, w, h):
    phi = angle[..., 0]
    theta = angle[..., 1]

    # when theta is out of bounds. Effectively, theta has rotated past the pole
    # so after adjusting theta to be in range, rotating phi by pi forms an
    # equivalent direction.
    theta = torch.clamp(fmod(theta, 2.0 * math.pi), 0.0, 2.0 * math.pi)
    out_bound_mask = (theta>math.pi)
    theta[out_bound_mask] = 2.0 * math.pi - theta[out_bound_mask]
    phi[out_bound_mask] += math.pi

    # Allow phi to repeat and map to the normal 0 to 2pi range.
    # Clamp and map after adjusting theta in case theta was forced to update phi.
    phi = torch.clamp(fmod(phi, 2.0 * math.pi), 0.0, 2.0 * math.pi)

    # Now phi is in [0, 2pi] and theta is in [0, pi] so it's simple to inverse
    # the linear equations in ImageCoordsToSphericalCoords, although there's no
    # -0.5 because we're returning floating point coordinates and so don't need
    # to center the pixel.
    return torch.stack([w * phi / (2.0 * math.pi), h * theta / math.pi], dim=-1).type_as(angle)

def _eval_legendre(l, m, x):

    '''
        // Evaluate the associated Legendre polynomial of degree @l and order @m at
        // coordinate @x. The inputs must satisfy:
        // 1. l >= 0
        // 2. 0 <= m <= l
        // 3. -1 <= x <= 1
        // See http://en.wikipedia.org/wiki/Associated_Legendre_polynomials
        //
        // This implementation is based off the approach described in [1],
        // instead of computing Pml(x) directly, Pmm(x) is computed. Pmm can be
        // lifted to Pmm+1 recursively until Pml is found
    '''

    # Compute Pmm(x) = (-1)^m(2m - 1)!!(1 - x^2)^(m/2), where !! is the double factorial.
    pmm = torch.ones_like(x)

    if m > 0:
        sign = (1 if m % 2 == 0 else -1)
        pmm = sign * double_factorial(2 * m - 1) * torch.pow(1 - x * x, m / 2.0)

    if l == m:
        # Pml is the same as Pmm so there's no lifting to higher bands needed
        return pmm

    # Compute Pmm+1(x) = x(2m + 1)Pmm(x)
    pmm1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        # Pml is the same as Pmm+1 so we are done as well
        return pmm1

    # Use the last two computed bands to lift up to the next band until l is
    # reached, using the recurrence relationship:
    # Pml(x) = (x(2l - 1)Pml-1 - (l + m - 1)Pml-2) / (l - m)
    for n in range(m+2, l+1):
        pmn = (x * (2 * n - 1) * pmm1 - (n + m - 1) * pmm) / (n - m)
        pmm = pmm1
        pmm1 = pmn

    return pmm1

def _eval_sh(l, m, angle=None, xyz=None):
    def _xyz2angle_z(xyz: torch.FloatTensor):
        # Z up
        return torch.stack([torch.atan2(xyz[:, 1], xyz[:, 0]), torch.acos(torch.clamp(xyz[:, 2], -1.0, 1.0))], dim=-1).type_as(xyz)
    def _angle_y2z(angle: torch.FloatTensor):
        return _xyz2angle_z(angle2xyz(angle))

    assert(l >= 0)
    assert(-l <= m and m <= l)
    assert((angle is None) != (xyz is None)) # only one of them should be filled

    # the original sh function uses z-axis as the top direction, while our method uses y-axis.
    # thus a convention of axes is required here.
    if angle is None:
        angle = _xyz2angle_z(xyz) 
    else:
        angle = _angle_y2z(angle)

    phi = angle[:, 0]
    theta = angle[:, 1]

    kml = math.sqrt((2.0 * l + 1) * factorial(l - abs(m)) / (4.0 * math.pi * factorial(l + abs(m))))

    if m > 0:
        return math.sqrt(2.0) * kml * torch.cos(m * phi) * _eval_legendre(l, m, torch.cos(theta))
    elif m < 0:
        return math.sqrt(2.0) * kml * torch.sin(-m * phi) * _eval_legendre(l, -m, torch.cos(theta))
    else: # m=0
        return kml * _eval_legendre(l, 0, torch.cos(theta))

def _eval_sh_sum(order, coeffs, angle=None, xyz=None):
    '''
        coeffs: [B, lm, 3]
        angle: [B, 3]
        xyz: [B, 3]
    '''
    if coeffs.ndim == 2:
        coeffs = coeffs[None,:,:]
    B = (angle.shape[0] if angle is not None else xyz.shape[0])
    sum = torch.zeros([B, coeffs.shape[-1]]).float().type_as(coeffs) # B, 3
    for l in range(order+1):
        for m in range(-l, l+1):
            sh = _eval_sh(l, m, angle, xyz) # B
            sum += sh[:, None] * coeffs[:,get_index(l, m)]
    return sum

def project_environment(order, env_img):
    assert(order>=0)
    h, w = env_img.shape[:2]
    pixel_area = (2.0 * math.pi / w) * (math.pi / h)
    coeffs = torch.zeros([get_coef_count(order), 3]).float().type_as(env_img)

    y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    coord = torch.stack([x, y], dim=-1).reshape(-1, 2).type_as(env_img)
    angle = imcoord2angle(coord, w, h)
    weight = pixel_area * torch.sin(angle[:, 1])
    color = env_img.reshape(-1, 3)
    for l in range(order+1):
        for m in range(-l, l+1):
            ind = get_index(l, m)
            sh = _eval_sh(l, m, angle)
            coeffs[ind] = ((sh * weight)[:,None] * color).sum(dim=0)
    return coeffs

def unproject_environment(order, coeffs, h, w, rand_noise=False, fast=False, pow_num=None, s=None):
    assert(order>=0)
    # assert(len(coeffs) == get_coef_count(order))

    y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    if rand_noise == True:
        x = x.type_as(coeffs)
        y = y.type_as(coeffs)
        x = (x+torch.rand_like(x)-0.5).clip(0, w)
        y = (y+torch.rand_like(y)-0.5).clip(0, h)
    coord = torch.stack([x, y], dim=-1).reshape(-1, 2).type_as(coeffs)

    angle = imcoord2angle(coord, w, h)

    img_recon = _eval_sh_sum(order, coeffs, angle) if not fast else fast_sh_sum(order, pow_num, s, coeffs, angle)
    img_recon = img_recon.reshape(h, w, 3)

    return img_recon




### -- Functions for SH rotation

# ---- The following functions are used to implement SH rotation computations
#      based on the recursive approach described in [1, 4]. The names of the
#      functions correspond with the notation used in [1, 4].
# See http://en.wikipedia.org/wiki/Kronecker_delta

kronecker_delta = lambda i, j: 1.0 if i == j else 0.0

def centered_elem(r, i, j):
    '''
        // [4] uses an odd convention of referring to the rows and columns using
        // centered indices, so the middle row and column are (0, 0) and the upper
        // left would have negative coordinates.
        //
        // This is a convenience function to allow us to access an Eigen::MatrixXd
        // in the same manner, assuming r is a (2l+1)x(2l+1) matrix.
    '''
    offset = (r.shape[1] - 1) // 2
    return r[:, i + offset, j + offset]
    
def P(i, a, b, l, r):
    '''
        // P is a helper function defined in [4] that is used by the functions U, V, W.
        // This should not be called on its own, as U, V, and W (and their coefficients)
        // select the appropriate matrix elements to access (arguments @a and @b).
    '''
    if b == l:
        return centered_elem(r[1], i, 1) * centered_elem(r[l - 1], a, l - 1) - centered_elem(r[1], i, -1) * centered_elem(r[l - 1], a, -l + 1)
    elif b == -l:
        return centered_elem(r[1], i, 1) * centered_elem(r[l - 1], a, -l + 1) + centered_elem(r[1], i, -1) * centered_elem(r[l - 1], a, l - 1)
    else:
        return centered_elem(r[1], i, 0) * centered_elem(r[l - 1], a, b)

# The functions U, V, and W should only be called if the correspondingly
# named coefficient u, v, w from the function ComputeUVWCoeff() is non-zero.
# When the coefficient is 0, these would attempt to access matrix elements that
# are out of bounds. The list of rotations, @r, must have the @l - 1
# previously completed band rotations. These functions are valid for l >= 2.

def U(m, n, l, r):
    '''
        // Although [1, 4] split U into three cases for m == 0, m < 0, m > 0
        // the actual values are the same for all three cases
    '''
    return P(0, m, n, l, r)

def V(m, n, l, r):
    if m == 0:
        return P(1, 1, n, l, r) + P(-1, -1, n, l, r)
    elif m > 0:
        return P(1, m - 1, n, l, r) * math.sqrt(1 + kronecker_delta(m, 1)) - P(-1, -m + 1, n, l, r) * (1 - kronecker_delta(m, 1))
    else:
        '''
            // Note there is apparent errata in [1,4,4b] dealing with this particular
            // case. [4b] writes it should be P*(1-d)+P*(1-d)^0.5
            // [1] writes it as P*(1+d)+P*(1-d)^0.5, but going through the math by hand,
            // you must have it as P*(1-d)+P*(1+d)^0.5 to form a 2^.5 term, which
            // parallels the case where m > 0.
        '''
        return P(1, m + 1, n, l, r) * (1 - kronecker_delta(m, -1)) + P(-1, -m - 1, n, l, r) * math.sqrt(1 + kronecker_delta(m, -1))

def W(m, n, l, r):
    if m == 0:
        # whenever this happens, w is also 0 so W can be anything
        return 0.0
    elif m > 0:
        return P(1, m + 1, n, l, r) + P(-1, -m - 1, n, l, r)
    else:
        return P(1, m - 1, n, l, r) - P(-1, -m + 1, n, l, r)

def compute_uvw(m, n, l):
    '''
        // Calculate the coefficients applied to the U, V, and W functions. Because
        // their equations share many common terms they are computed simultaneously.
    '''
    d = kronecker_delta(m, 0)
    denom = (2.0 * l * (2.0 * l - 1) if abs(n) == l else (l + n) * (l - n))

    u = math.sqrt((l + m) * (l - m) / denom)
    v = 0.5 * math.sqrt((1 + d) * (l + abs(m) - 1.0) * (l + abs(m)) / denom) * (1 - 2 * d)
    w = -0.5 * math.sqrt((l - abs(m) - 1) * (l - abs(m)) / denom) * (1 - d)
    return u,v,w

def compute_band_rotation(l, rotations: list):
    '''
        // Calculate the (2l+1)x(2l+1) rotation matrix for the band @l.
        // This uses the matrices computed for band 1 and band l-1 to compute the
        // matrix for band l. @rotations must contain the previously computed l-1
        // rotation matrices, and the new matrix for band l will be appended to it.
        //
        // This implementation comes from p. 5 (6346), Table 1 and 2 in [4] taking
        // into account the corrections from [4b].
        
        // The band's rotation matrix has rows and columns equal to the number of
        // coefficients within that band (-l <= m <= l implies 2l + 1 coefficients).
    '''
    rotation = torch.zeros(rotations[0].shape[0], 2*l+1, 2*l+1).float().type_as(rotations[0])
    for m in range(-l, l+1):
        for n in range(-l, l+1):
            u, v, w = compute_uvw(m, n, l)
            if not nearbymargin(u, 0):
                u *= U(m, n, l, rotations)
            if not nearbymargin(v, 0):
                v *= V(m, n, l, rotations)
            if not nearbymargin(w, 0):
                w *= W(m, n, l, rotations)

            rotation[:, m+l, n+l] = u + v + w
    rotations.append(rotation)

# For efficiency, the cosine lobe for normal = (0, 0, 1) as the first 9
# spherical harmonic coefficients are hardcoded below. This was computed by
# evaluating:
#   ProjectFunction(kIrradianceOrder, [] (double phi, double theta) {
#     return Clamp(Eigen::Vector3d::UnitZ().dot(ToVector(phi, theta)), 
#                  0.0, 1.0);
#   }, 10000000);
def create_rot_by_mat(order, R_mat):
    ret = []

    # l = 0
    ret.append(torch.ones([R_mat.shape[0], 1,1]).type_as(R_mat))

    # l = 1
    r = torch.stack([R_mat[:, 1, 1], -R_mat[:, 1, 2], R_mat[:, 1, 0], 
                    -R_mat[:, 2, 1], R_mat[:, 2, 2], -R_mat[:, 2, 0], 
                    R_mat[:, 0, 1], -R_mat[:, 0, 2], R_mat[:, 0, 0]], dim=-1).reshape(-1, 3, 3)
    ret.append(r)

    for l in range(2, order+1):
        compute_band_rotation(l, ret)
    
    return ret

def rotate_coeff_by_rotmat(order, R_mat, coeff):
    '''
        input:
            order: int
            R_mat: tensor of shape [B1, 3, 3]
            coeff: tensor of shape [B2, N]
        return:
            tensor of shape [B1*B2, N]
    '''
    R_list = create_rot_by_mat(order, R_mat)

    # transform one band(order) at a time
    # equivalent to a huge matrix multipilication, 
    batch_rot = R_mat.shape[0]
    batch_coeff = coeff.shape[0]
    batch = batch_rot * batch_coeff
    ret = coeff[None,:,:].repeat([batch_rot, 1, 1]).reshape(batch, -1)
    for l in range(order+1):
        band_coeff = coeff[None, :, l*l:(l+1)*(l+1), None].repeat([batch_rot, 1, 1, 1]).reshape(batch, 2*l+1, 1) # B x 2l+1 x 1
        rot_coeff = R_list[l][:, None, :, :].repeat(1, batch_coeff, 1, 1).reshape(batch, 2*l+1, 2*l+1) # B x 2l+1 x 2l+1
        band_coeff = torch.bmm(rot_coeff, band_coeff) # B x 2l+1 x 1
        ret[:, l*l:(l+1)*(l+1)] = band_coeff[:, :, 0]
    
    return ret

def rotate_coeff_by_normal(order, normal, coeff):
    normal = normal / (normal.norm(dim=1, keepdim=True)+1e-9)
    z = torch.zeros_like(normal)
    z[:, 2] = 1 
    x = torch.cross(normal, z)
    x = x / (x.norm(dim=1, keepdim=True)+1e-9)
    y = torch.cross(normal, x)
    R_mat = torch.stack([x, y, normal], dim=-1)
    I_mat = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).float().type_as(R_mat)
    I_mat_2 = torch.from_numpy(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])).float().type_as(R_mat)
    R_mat[normal[:,2]>0.999] = I_mat
    R_mat[normal[:,2]<-0.999] = I_mat_2            
    return rotate_coeff_by_rotmat(order, R_mat, coeff)

def rotate_coeffs(orders, R_mat, coeffs):    
    rotated_coeffs = rotate_coeff_by_rotmat(orders, R_mat, coeffs.transpose(1, 0))
    rotated_coeffs = rotated_coeffs.reshape(-1, 3, get_coef_count(orders)).transpose(1, 2)
    return rotated_coeffs




### -- Functions for irradiance calculation
### the irradiance is normalized by 1/pi

def render_irrandiance_rotate(coeffs, normal):
    '''Calculate irradiance using method using SH rotation matrix'''
    cosine_lobe = torch.FloatTensor([0.886227, 0.0, 1.02333, 0.0, 0.0, 0.0, 0.495416, 0.0, 0.0]).type_as(normal)[None,:] # 1 x 9
    rotation = rotate_coeff_by_normal(2, normal, cosine_lobe) / math.pi # N x 9
    #print("!!!!", rotation.mean().item())
    return (coeffs[:, :9, :] * rotation[:, :, None]).sum(dim=1)

def render_irrandiance_sh_sum(coeffs, normal, fast=False, pow_num=None, s=None):
    '''Calculate irradiance using method proposed by Ramamoorthi et al., 2001'''
    cosine_lobe = torch.FloatTensor([3.14, 2.09, 2.09, 2.09, 0.79, 0.79, 0.79, 0.79, 0.79]).type_as(normal)[None,:] # 1 x 9
    coeffs = coeffs[:, :9, :] * cosine_lobe[:, :, None] / math.pi
    return _eval_sh_sum(2, coeffs, xyz=normal) if not fast else fast_sh_sum(2, pow_num, s, coeffs, xyz=normal)
    
def render_irrandiance_map_rotate(coeffs, h, w):
    '''Calculate irradiance map using method using SH rotation matrix'''
    y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    coord = torch.stack([x, y], dim=-1).reshape(-1, 2).type_as(coeffs)
    angle = imcoord2angle(coord, w, h)
    return render_irrandiance_rotate(coeffs[None,:,:], angle2xyz(angle)).reshape(h, w, 3)

def render_irrandiance_map_sh_sum(coeffs, h, w, fast=False, pow_num=None, s=None):
    '''Calculate irradiance map using method proposed by Ramamoorthi et al., 2001'''
    y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    coord = torch.stack([x, y], dim=-1).reshape(-1, 2).type_as(coeffs)
    angle = imcoord2angle(coord, w, h)
    return render_irrandiance_sh_sum(coeffs[None,:,:], angle2xyz(angle), 
                                                fast=fast, pow_num=pow_num, s=s).reshape(h, w, 3)
    
def render_irrandiance_map_direct(env_map, outh, outw):
    '''Calculate irradiance map directly from environment map'''
    import tqdm
    h, w = env_map.shape[:2] 
    pixel_area = 2 * math.pi / w * math.pi/ h
    return_img = torch.zeros([outh, outw, 3], device=env_map.device)
    y, x = torch.meshgrid(torch.arange(0, outh), torch.arange(0, outw))
    angle = torch.stack([(x + 0.5) * 2 * math.pi / outw, (y + 0.5) * math.pi / outh], dim=-1).type_as(env_map) # h, w, 2
    normal = angle2xyz(angle)
    for ey in tqdm.tqdm(range(h)):
        theta = torch.FloatTensor([(ey + 0.5) * math.pi / h]).type_as(env_map)
        sa = pixel_area * torch.sin(theta)
        for ex in range(w):
            light_dir = torch.FloatTensor([(ex + 0.5) * 2 * math.pi / w,(ey + 0.5) * math.pi / h]).type_as(env_map) # 2
            light_dir = angle2xyz(light_dir[None, None,:])
            light = env_map[ey,ex] * sa
            return_img += light[None,None,:] / math.pi * (light_dir*normal).sum(-1, keepdim=True).clamp(0.0, 1.0)
    return return_img




### -- Functions for phong BRDF rendering

def _render_phong_sh_sum(orders, coeffs, refl_dir, fast=False, pow_num=None, s=None):
    B_coef = torch.arange(0, coeffs.shape[1], device=coeffs.device)[None,:,None] # 1, N_sh, 1
    B_coef = torch.pow(B_coef, 0.5).floor()
    glossiness = 32
    B_coef = torch.exp(-B_coef * B_coef / 2 / glossiness) * coeffs # torch.pow(4*3.14159 / (2*B_coef+1), 0.5) * torch.exp(-B_coef / 2 / s / s) * coeffs
    # B_coef = coeffs
    return _eval_sh_sum(orders, B_coef, xyz=refl_dir) if not fast else fast_sh_sum(orders, pow_num, s, B_coef, xyz=refl_dir)  # N_rays*N_samples, 3
    
def _render_phong_map_sh_sum(order, coeffs, h, w, fast=False, pow_num=None, s=None):
    y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    coord = torch.stack([x, y], dim=-1).reshape(-1, 2).type_as(coeffs)
    angle = imcoord2angle(coord, w, h)
    if coeffs.ndim == 2:
        coeffs = coeffs[None,:,:]
    return _render_phong_sh_sum(order, coeffs, angle2xyz(angle), fast, pow_num, s).reshape(h, w, 3)
 
def _compute_phong_direct(env_map, outh, outw, s = 128):
    import tqdm
    h, w = env_map.shape[:2] 
    pixel_area = 2 * math.pi / w * math.pi/ h
    return_img = torch.zeros([outh, outw, 3], device=env_map.device)
    y, x = torch.meshgrid(torch.arange(0, outh), torch.arange(0, outw))
    normal = torch.stack([(x + 0.5) * 2 * math.pi / outw, (y + 0.5) * math.pi / outh], dim=-1).type_as(env_map) # h, w, 2
    normal = angle2xyz(normal)
    for ey in tqdm.tqdm(range(h)):
        theta = torch.FloatTensor([(ey + 0.5) * math.pi / h]).type_as(env_map)
        sa = pixel_area * torch.sin(theta)
        for ex in range(w):
            light_dir = torch.FloatTensor([(ex + 0.5) * 2 * math.pi / w,(ey + 0.5) * math.pi / h]).type_as(env_map) # 2
            light_dir = angle2xyz(light_dir[None, None,:])
            light = env_map[ey,ex] * sa
            return_img += light[None,None,:] * (s+1)/2/math.pi * torch.pow((light_dir*normal).sum(-1, keepdim=True).clamp(0.0, 1.0), s)
    return return_img

def _render_ball(light_map, img_res, is_reflect=False):
    y, x = torch.meshgrid(torch.arange(0, img_res), torch.arange(0, img_res))
    y = -(y-img_res / 2) / img_res * 2
    x = (x-img_res/2) / img_res * 2
    z = 1-y*y-x*x
    mask = (z < 0).reshape(-1)
    normal = torch.stack([x,y,z],dim=-1) # H x W x 3
    fov = 20/180*math.pi
    d = math.tan(fov/2)
    view_dir = normalize(torch.stack([-x*d, -y*d, z*0+1], dim=-1)) # H * W * 3
    if is_reflect == False:
        dir = normal 
    else:
        mask += (view_dir*normal).reshape(-1, 3).sum(dim=-1)<0
        dir = normalize(2 * (view_dir*normal).sum(dim=-1, keepdim=True)*normal - view_dir)

    coord = xyz2angle(dir.reshape(-1, 3)) # HW x 2
    #coord[:,0] += math.pi/2
    coord = torch.where(coord<0, coord+math.pi*2, coord)
    coord = torch.stack([coord[:,1]/math.pi*light_map.shape[0]-0.5, coord[:,0]/math.pi/2*light_map.shape[1]-0.5], dim=-1).long() # HW x 2
    ret_img = light_map[coord[:,0].clamp(0, light_map.shape[0]-1), coord[:,1].clamp(0, light_map.shape[1]-1)]
    ret_img[mask==1] = -1
    ret_img = ret_img.reshape(img_res, img_res, 3)
    return ret_img