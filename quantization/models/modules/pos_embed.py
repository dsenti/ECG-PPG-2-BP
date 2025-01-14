# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np
import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------





# def get_keep_chans_pos_enc(emb_dim, nr_channels, nr_patches_H, nr_patches_W, device, cls_token=False):
#     assert emb_dim % 3 == 0, 'embedding dimension (`emb_dim`) is not divisible by 2; cannot create equal split between chans and patches pos enc encodings'
#     emb_dim_chans = emb_dim // 2
#     emb_dim_patches = emb_dim // 3
#     W_patch_enc = get_patches_enc(emb_dim_patches, nr_channels, nr_patches_W, device) #C, W, D
#     H_patch_enc = get_patches_enc(emb_dim_chans, nr_channels, nr_patches_H, device) #C, H, D
#     chans_enc = get_channels_enc(emb_dim_chans, nr_channels, nr_patches_W*nr_patches_H, device) # C, H*W, D
    
#     encodings = torch.concatenate([chans_enc, patches_enc], axis=-1)
    
#     if cls_token:
#         encodings = np.concatenate([np.zeros([1, embed_dim]), encodings], axis=0)

#     return encodings    

def get_throw_chans_pos_enc(emb_dim, seq_len, device, cls_token=False):
    encodings = get_patches_enc(emb_dim, seq_len, device)
    if cls_token:
        encodings = np.concatenate([np.zeros([1, emb_dim]), encodings], axis=0)
    return encodings    

def get_keep_chans_pos_enc(emb_dim, nr_channels, nr_patches_per_chan, device, cls_token=False):
    assert emb_dim % 2 == 0, 'embedding dimension (`emb_dim`) is not divisible by 2; cannot create equal split between chans and patches pos enc encodings'
    emb_dim_chans = emb_dim // 2
    emb_dim_patches = emb_dim // 2
    patches_enc = get_patches_enc(emb_dim_patches, nr_channels, nr_patches_per_chan, device)
    chans_enc = get_channels_enc(emb_dim_chans, nr_channels, nr_patches_per_chan, device)
    encodings = torch.concatenate([chans_enc, patches_enc], axis=-1) # nr chans, nr_patches_per_chan, D_c
    encodings = encodings.reshape(-1, emb_dim) # nr_chans*nr_patches, D_c
    
    if cls_token:
        encodings = np.concatenate([np.zeros([1, emb_dim]), encodings], axis=0)

    return encodings    

def get_patches_enc(emb_dim_patches, nr_channels, nr_patches_per_chan, device):
    patches_enc = get_pos_enc(emb_dim_patches, nr_patches_per_chan, device) # nr_chans, D_c
    patches_enc = patches_enc.unsqueeze(0)
    patches_enc = patches_enc.repeat(nr_channels, 1, 1) # nr_chans, nr_patches_per_chan, D_c
    return patches_enc

def get_channels_enc(emb_dim_chans, nr_channels, nr_patches_per_chan, device):
    chans_enc = get_pos_enc(emb_dim_chans, nr_channels, device) # nr_chans, D_c
    chans_enc = chans_enc.unsqueeze(1)
    chans_enc = chans_enc.repeat(1, nr_patches_per_chan, 1) # nr_chans, nr_patches_per_chan, D_c
    return chans_enc
    

def get_pos_enc(emb_dim, seq_len, device, alternating_sincos=False):
    # same size with input matrix (for adding with input matrix)
    encoding = torch.zeros(seq_len, emb_dim, device=device)
    encoding.requires_grad = False  # we don't need to compute gradient
    
    pos = torch.arange(0, seq_len, device=device)
    pos = pos.float().unsqueeze(dim=1)
    
    if alternating_sincos:
        # 1D => 2D unsqueeze to represent word's position
        _2i = torch.arange(0, emb_dim, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        
        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / emb_dim)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / emb_dim)))
    else:
        # 1D => 2D unsqueeze to represent word's position
        first_half = emb_dim//2
        _i = torch.arange(0, first_half, step=1, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        
        encoding[:, :first_half] = torch.sin(pos / (10000 ** (_i / emb_dim)))
        encoding[:, first_half:] = torch.cos(pos / (10000 ** (_i / emb_dim)))
        
    # compute positional encoding to consider positional information of words
    return encoding[:seq_len, :]
    
def get_3d_sincos_pos_embed(embed_dim, grid_size, num_channels, cls_token=False):
    """
    Generate 3D sine-cosine positional embeddings.
    grid_size: int, height and width of the grid
    num_channels: int, number of channels
    return:
    pos_embed: [grid_size*grid_size*num_channels, embed_dim] or [1+grid_size*grid_size*num_channels, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_c = np.arange(num_channels, dtype=np.float32)
    
    # Create a 3D grid with c, h, and w (change order to match patch embedding output)
    grid = np.meshgrid(grid_c, grid_h, grid_w, indexing='ij')
    grid = np.stack(grid, axis=0)  # Shape: [3, num_channels, grid_size, grid_size]
    
    # Reshape to [3, 1, num_channels, grid_size, grid_size]
    grid = grid.reshape([3, 1, num_channels, grid_size, grid_size])
    # print(grid.shape)
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = pos_embed.reshape(num_channels * grid_size * grid_size, embed_dim)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    print(f"pos_embed shap before return: {pos_embed.shape}")
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size_height, grid_size_width, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size_height, dtype=np.float32)
    grid_w = np.arange(grid_size_width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_height, grid_size_width])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0
    # Encode each of the 3 dimensions separately and concatenate
    emb_c = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # Shape: [C*H*W, D/3]
    print(emb_c.shape)
    print_embc = emb_c.reshape(23, -1, 768//3)
    print('emb_c', np.unique(print_embc[0]).shape)
    print('emb_c[1]', np.unique(print_embc[1]).shape)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # Shape: [C*H*W, D/3]
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # Shape: [C*H*W, D/3]
    emb = np.concatenate([emb_c, emb_h, emb_w], axis=1)  # Shape: [C*H*W, D]
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    print(grid.shape)
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    print('emb_h.shape', emb_h.shape)
    print('emb_w.shape', emb_w.shape)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb




# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
            
            
            
# embed_dim = 768
# grid_size_height = 8
# grid_size_width = 8
# grid_size = 8
# num_channels = 23




# # # get_2d_sincos_pos_embed(embed_dim, grid_size_height, grid_size_width)
# # grid_h = np.arange(grid_size, dtype=np.float32)
# # grid_w = np.arange(grid_size, dtype=np.float32)
# # grid_c = np.arange(num_channels, dtype=np.float32)

# # # Create a 3D grid with c, h, and w (change order to match patch embedding output)
# # grid = np.meshgrid(grid_c, grid_h, grid_w, indexing='ij')
# # grid = np.stack(grid, axis=0)  # Shape: [3, num_channels, grid_size, grid_size]

# # # Reshape to [3, 1, num_channels, grid_size, grid_size]
# # grid = grid.reshape([3, 1, num_channels, grid_size, grid_size])
# # # print(grid.shape)
# # pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
# # pos_embed = pos_embed.reshape(num_channels * grid_size * grid_size, embed_dim)
# pos_emb = get_2d_sincos_pos_embed(embed_dim, grid_size_height=20, grid_size_width=23, cls_token=False)


# # grid_h = np.arange(grid_size, dtype=np.float32)
# # emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)