import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

# NeROIC Model for Geometry/Rendering Network
class NeROIC(nn.Module):
    def __init__(self, args, model_type, 
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
                 beta_min=0.03):
        """
        ---Parameters for the original NeRF---
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)

        ---Parameters for NeRF-W (used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        in_channels_t: transient embedding dimension. n^(tau) in the paper
        beta_min: minimum pixel color variance

        --Parameters for NeROIC---
        model_type: type of the network (geometry or rendering)
        """
        super().__init__()

        assert(model_type in ["geometry", "rendering"])

        self.args = args
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir

        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.in_channels_t = in_channels_t if encode_transient else 0
        self.beta_min = beta_min

        self.model_type = model_type

        # -- COMMON LAYERS --
        # position encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                    nn.Linear(W+in_channels_dir+self.in_channels_a, W//2), nn.ReLU(True))        

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        # static beta is only used for transient-free model
        self.static_beta = nn.Sequential(nn.Linear(W+self.in_channels_a, 1), nn.Softplus())

        # -- GEOMETRY NETWORK LAYERS--
        if self.model_type == "geometry":
            self.static_rgb = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
            if encode_transient:
                # transient encoding layers
                self.transient_encoding = nn.Sequential(
                                            nn.Linear(W+in_channels_t, W//2), nn.ReLU(True),
                                            nn.Linear(W//2, W//2), nn.ReLU(True),
                                            nn.Linear(W//2, W//2), nn.ReLU(True),
                                            nn.Linear(W//2, W//2), nn.ReLU(True))
                # transient output layers
                self.transient_sigma = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
                self.transient_beta = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
                self.transient_rgb = nn.Sequential(nn.Linear(W//2, 3))

        # -- RENDERING NETWORK LAYERS --
        if self.model_type == "rendering":
            self.material_encoding = []
            for i in range(4):
                if i == 0:
                    layer = nn.Linear(in_channels_xyz+W, W//2)
                else:
                    layer = nn.Linear(W//2, W//2)
                self.material_encoding.extend([layer, nn.ReLU(True)])
            self.material_encoding = nn.Sequential(*self.material_encoding)

            self.static_albedo = nn.Sequential(nn.Linear(W//2, 3), nn.Sigmoid())
            self.static_normal = nn.Sequential(nn.Linear(W//2, 3))
            self.static_specular = nn.Sequential(nn.Linear(W//2, 1), nn.Sigmoid())
            self.static_glossiness = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())

            if encode_transient:
                # adding separate transient layers for rendering network 
                # to prevent over-writting from geometry network
                self.transient_encoding_rendering = nn.Sequential(
                                            nn.Linear(W+in_channels_t, W//2), nn.ReLU(True),
                                            nn.Linear(W//2, W//2), nn.ReLU(True),
                                            nn.Linear(W//2, W//2), nn.ReLU(True),
                                            nn.Linear(W//2, W//2), nn.ReLU(True))

                self.transient_sigma_rendering = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
                self.transient_beta_rendering = nn.Sequential(nn.Linear(W//2, 1), nn.Softplus())
                self.transient_rgb_rendering = nn.Sequential(nn.Linear(W//2, 3))



    def forward(self, x, freeze_geometry, sigma_only=False, output_transient=True):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: the embedded vector of position (+ direction + appearance + transient)
            freeze_geometry: whether to freeze geometry network.
            sigma_only: whether to infer sigma only.
            output_transient: whether to infer the transient component.

        Outputs (concatenated):
            if sigma_only:
                static_sigma
            elif output_transient:
                static_rgb, static_sigma, transient_rgb, transient_sigma, transient_beta
            else:
                static_rgb, static_sigma
        """

        if sigma_only:
            input_xyz = x
        elif output_transient:
            input_xyz, input_dir_a, input_t = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a,
                                self.in_channels_t], dim=-1)
        else:
            input_xyz, input_dir_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir+self.in_channels_a], dim=-1)
            

        xyz_ = input_xyz # (B, d_xyz)
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        # geometry output
        static_sigma = self.static_sigma(xyz_) # (B, 1)
        if self.model_type == "rendering" or freeze_geometry: # don't train geometry
            xyz_ = xyz_.detach() # (B, W)
            static_sigma = static_sigma.detach() # (B, 1)
        if sigma_only:
            return static_sigma 

        xyz_encoding_final = self.xyz_encoding_final(xyz_) # (B, W)

        # rgb/material output
        if self.model_type == "geometry":
            dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
            dir_encoding = self.dir_encoding(dir_encoding_input) # (B, W)
            static_rgb = self.static_rgb(dir_encoding) # (B, 3)
        else:
            material_encoding_input = torch.cat([xyz_, input_xyz], 1)
            material_encoding = self.material_encoding(material_encoding_input)
            
            static_albedo = self.static_albedo(material_encoding) # (B, 3)
            static_normal = normalize(self.static_normal(material_encoding)) # (B, 3)
            static_specular = self.static_specular(material_encoding) # (B, 1)
            static_glossiness = self.static_glossiness(material_encoding) # (B, 1)
            
            static_rgb = torch.cat([static_albedo, static_normal, static_specular, static_glossiness], 1) # (B, 8)
            
        static = torch.cat([static_rgb, static_sigma], 1) # (B, 4 or 9)

        if not output_transient:
            input_a = input_dir_a[...,self.in_channels_dir:]
            static_beta = self.static_beta(torch.cat([xyz_encoding_final, input_a], 1))
            static = torch.cat([static, static_beta], 1) # (B, 5 or 10)
            return static

        # transient outputs
        if self.model_type == "geometry":
            transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
            transient_encoding = self.transient_encoding(transient_encoding_input)
            transient_sigma = self.transient_sigma(transient_encoding) # (B, 1)
            transient_rgb = torch.sigmoid(self.transient_rgb(transient_encoding)) # (B, 3)
            transient_beta = self.transient_beta(transient_encoding) # (B, 1)
        else:
            transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
            transient_encoding = self.transient_encoding_rendering(transient_encoding_input)
            transient_sigma = self.transient_sigma_rendering(transient_encoding) # (B, 1)
            transient_rgb = torch.sigmoid(self.transient_rgb_rendering(transient_encoding)) # (B, 3)
            transient_beta = self.transient_beta_rendering(transient_encoding) # (B, 1)
            
        transient = torch.cat([transient_rgb, transient_sigma, transient_beta], 1) # (B, 5)
        
        return torch.cat([static, transient], 1) # (B, 9 or 14)