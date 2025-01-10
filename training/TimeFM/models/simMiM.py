# Copyright (c) Insitro, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.d
import math
from functools import partial
from typing import List
import numpy as np
import random
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from models.modules.attention import CustomAttentionBlock
from models.modules.pos_embed import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid, get_keep_chans_pos_enc, get_throw_chans_pos_enc
from models.modules.patching import PatchingModule



def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

def random_masking(x, mask_ratio, attn_mask=None):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [B = batch_size, N = num_tokens, D = embed_dim]
    attn_mask: [B, N]
    """
    B, N, D = x.shape  # N = batch_size, N = num_tokens, D = embed_dim
    if attn_mask is not None:
        nr_padded_tokens = attn_mask.shape[-1] - attn_mask.sum(axis=-1) # (B)
        mask_keep = ((N - nr_padded_tokens) * (1-mask_ratio)).to(int) # (B)
    else:
        len_keep = int(N * (1 - mask_ratio)) # int
    
    
    noise = torch.rand(B, N, device=x.device)  # noise in [0, 1] of shape (B,N)

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove; shape (B,N)
    ids_restore = torch.argsort(ids_shuffle, dim=1) # shape (B,N)

    # get total nr of tokens padded per batch
    # keep the first subset
    if attn_mask is not None:
        keep_ids = ids_shuffle < mask_keep.unsqueeze(1) # shape (B,N)
        x_shuffled = torch.gather(x, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, D)) # shape (B,len_keep,D)
        x_masked = torch.zeros_like(x) # shape (B,N,D)
        x_masked[keep_ids] = x_shuffled[keep_ids] 
    else:
        ids_keep = ids_shuffle[:, :len_keep] # shape (B,len_keep)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # shape (B,len_keep,D)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([B, N], device=x.device) # shape (B,N)
    if attn_mask is not None:
        mask[keep_ids] = 0 # shape (B,N)
    else:
        mask[:, :len_keep] = 0 # shape (B,len_keep)

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore) # shape (B,N,D)

    return x_masked, mask, ids_restore # # shapes (B,len_keep,D), (B,N), (B,N) 


class ChannelVisionTransformer(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=23,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, masking_ratio=0.75, keep_chans=True, using_spectrogram=True, learned_enc=True,
                 attention_type='default', square_patches=False, drop_path=0.0):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.masking_ratio = masking_ratio
        self.in_chans = in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.keep_chans = keep_chans
        self.square_patches = square_patches
        self.patch_embed = PatchingModule(img_size, patch_size, in_chans, embed_dim, keep_chans=keep_chans, using_spectrogram=using_spectrogram, square_patches=self.square_patches)
        self.num_patches = self.patch_embed.num_patches
        self.patches_per_dim = self.img_size // self.patch_size
        self.using_spectrogram = using_spectrogram
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pad_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.learned_enc = learned_enc
        if learned_enc:
            if self.keep_chans:
                self.channel_encoding = nn.Parameter(torch.zeros(1, 64,self.embed_dim//2))
                self.patch_encoding = nn.Parameter(torch.zeros(1, 100,self.embed_dim//2))
            else:
                self.pos_encoding = nn.Parameter(torch.zeros(1,3000,self.embed_dim))

        self.blocks = nn.ModuleList([
            CustomAttentionBlock(dim=self.embed_dim, num_channels=in_chans, num_heads=self.num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, attention_type=attention_type, block_idx=i, drop_path=drop_path)
            for i in range(depth)])
        self.norm = norm_layer(self.embed_dim)
        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        # Encodings Initializations code taken from the LaBraM paper
        trunc_normal_(self.pad_token, std=0.02)
        trunc_normal_(self.mask_token, std=0.02)
        
        if self.learned_enc:
            if self.keep_chans:
                trunc_normal_(self.channel_encoding, std=0.02)
                trunc_normal_(self.patch_encoding, std=0.02)
            else:
                trunc_normal_(self.pos_encoding, std=0.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
                    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
                      
    def add_pos_encodings(self, x, nr_chans, add_cls_token=False):
        if self.learned_enc:
            return self.add_learned_pos_encodings(x, nr_chans, add_cls_token)
        else:
            return self.add_static_pos_encodings(x, nr_chans, add_cls_token)
        
    def add_static_pos_encodings(self, x, nr_chans, add_cls_token=False):
        B, N, D = x.shape
        patches_per_chan = N // nr_chans #int
        if self.using_spectrogram and self.square_patches:
            pos_embed = get_3d_sincos_pos_embed(embed_dim=self.embed_dim, grid_size=self.patches_per_dim, num_channels=nr_chans, cls_token=add_cls_token)            
        elif self.keep_chans:
            pos_embed = get_keep_chans_pos_enc(D, nr_chans, patches_per_chan, device=x.device, cls_token=add_cls_token)
        else:
            pos_embed = get_throw_chans_pos_enc(D, N, device=x.device, cls_token=add_cls_token)
        # pos embed in all cases: (C, D)
        pos_embed = pos_embed.unsqueeze(0) # (1, N, D)
        return x + pos_embed # (B, N, D)
    
    def add_learned_pos_encodings(self,x, nr_chans, add_cls_token=False):
            B, N, D = x.shape
            
            if self.keep_chans:
                patches_per_chan = N // nr_chans # int
                chan_enc = self.channel_encoding[:, :nr_chans, :] # (1, C, D//2)

                patch_enc = self.patch_encoding[:, :patches_per_chan, :] # (1, C, D//2)
                
                chan_enc = chan_enc.unsqueeze(2).repeat(1, 1, patches_per_chan, 1) # (1, C, P, D//2) where P = nr patches/tokens per channel
                patch_enc = patch_enc.unsqueeze(1).repeat(1, nr_chans, 1, 1) # (1, C, P, D//2)
                pos_enc = torch.concatenate([chan_enc, patch_enc], axis=-1) # (1, C, P, D)
                pos_enc = pos_enc.reshape((1, N, D)) # (1, N, D)
            else:
                pos_enc = self.pos_encoding[:, :N, :] # (1, N, D)
            
            return x + pos_enc # (B, C, D)
    
    def prepare_tokens(self, x, attn_mask=None):
        B = x.shape[0] # (B, N, D)
        x, mask, ids_restore = random_masking(x, self.masking_ratio, attn_mask) # (B,len_masked,D), (B,N), (B,N)

        # append mask tokens to sequence and unshuffle
        nr_mask_tokens = ids_restore.shape[1] - x.shape[1] # int
        mask_tokens = self.mask_token.repeat(B, nr_mask_tokens, 1) # (B masked_tokens, D)
        x = torch.cat([x, mask_tokens], dim=1) # (B, N, D)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # (B, N, D)        
        return x, mask, ids_restore # (B,N,D), (B,N), (B,N)
    
    def forward(self, x, nr_channels_padded=None, mask_tokens=True): #mask_tokens = True if doing MAE pre-training. False for fine-tuning.
        """
        Args:
            x: Input waveform / spectrogram / Image / Timeseries
            nr_channels_padded: array telling how many channels were padded. (size B, C) 
            pad_channels: True if channels were padded, Defaults to False.
            mask_tokens: Defaults to True.

        Returns:
            _type_: _description_
        """
        # Patch embedding
        B, C = x.shape[:2] # B = batch size; C = num channels;  = Sequence length 
        x = self.patch_embed(x)  # Shape: [B, num tokens N, embed_dim D]

        B, N, D = x.shape 
        P = N // C # P = num per-channel tokens (i.e. patches)

        attn_mask = None
        
        if nr_channels_padded is not None:
            # since we have a varying number of padded channels in each batch, we take the approach
            # of creating an index mask, thus assigning the [PAD] token value to each to-be-padded location
            # NOTE: each to-be-padded location is expected to be already 0-padded in the dataset class,
            #       to avoid array stacking errors
            nr_real_chans = C - nr_channels_padded # vector of size (B, C)            
            channel_indices = torch.arange(C).unsqueeze(0).to(x.device)  # Shape (1, C)
            pad_mask = channel_indices >= nr_real_chans.unsqueeze(1)  # Shape (B, C)
            pad_mask = pad_mask.repeat_interleave(P, dim=1) # Shape (B, C*P)
            
            # extract attn_mask of shape B C*P and assign 0 to padded tokens, 1 to non-padded tokens
            attn_mask = (~pad_mask).int() # Shape (B, C*P)
            
            # expand pad mask and assign learned [PAD] to input
            pad_mask = pad_mask.unsqueeze(-1).expand(-1, -1, D)  # Shape (B, C*P, D)
            x[pad_mask] = self.pad_token.expand(B, C*P, -1)[pad_mask] # x remains at the same shape (B, N, D), only the padded tokens are modified
            
        if mask_tokens:
            # mask, unshuffle and get ids of tokens to be restored
            x, mask, ids_restore = self.prepare_tokens(x, attn_mask) # (B, N, D)
            
        # add pos encodings
        x = self.add_pos_encodings(x, C, add_cls_token=False) # (B, N, D)

        # forward pass through transformer blocks
        for blk in self.blocks:
            x = blk(x, attn_mask) # (B, N, D)
        x = self.norm(x) # (B, N, D)
        
        # return output according to whether pretraining (mask_tokens) or finetuning 
        if mask_tokens:
            # x:(B, N, D)
            # mask, ids_restore, attn_mask: (B, N)
            return x, mask, ids_restore, attn_mask 
        else:
            return x # (B, N, D)

    