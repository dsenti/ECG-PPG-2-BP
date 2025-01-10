import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from typing import Optional
from torch.jit import Final
#from timm.models.vision_transformer import Block
from timm.layers import Mlp, DropPath, use_fused_attn
import pdb


class AlternatingAttention(nn.Module):
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            in_chans = 38, #23,
            block_idx   = 0
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False # use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.do_channel_attn = block_idx % 2 == 0
        self.in_chans = in_chans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        T = N // self.in_chans

        if self.do_channel_attn:
            # If the current block is a channel attention block, perform attention over the Channels dimension C
            x = rearrange(x, 'B (C T) D -> (B T) C D', C=self.in_chans)
            x = self.forward_attn(x)
            x = rearrange(x, '(B T) C D -> B (C T) D', T=T)
        else:
            # If the current block is a patch-attention block, perform attention over the patches dimension T
            x = rearrange(x, 'B (C T) D -> (B C) T D', C=self.in_chans)
            x = self.forward_attn(x)
            x = rearrange(x, '(B C) T D -> B (C T) D', C=self.in_chans)
        return x
    
    def forward_attn(self, x):
        '''
        Default attention mechanism
        '''
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BottleneckSplitAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        qk_norm: bool = False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        nr_channels=23
    ):
        super(BottleneckSplitAttention, self).__init__()
        self.nr_channels=nr_channels
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        
        B, N, D = x.shape
        C = self.nr_channels
        t = N // C
        
        # qkv projection and norms
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        B, H, N, d = q.shape
        
        # preparation of tokens for bottleneck attention
        q = q.view(B, H, C, t, d) 
        k = k.view(B, H, C, t, d) 
        v = v.view(B, H, C, t, d) 
        
        # Attention over Channels dimension
        v = self.qkv_attn(
            q.mean(-2).squeeze(-2), # #B H C d
            k.mean(-2).squeeze(-2), # #B H C d
            v.mean(-2).squeeze(-2), # #B H C d
            ) 
        
        # token+channel mixed attention
        q = q.transpose(2, 3).mean(-2).squeeze(-2)  #B H t d
        k = k.transpose(2, 3).mean(-2).squeeze(-2)  #B H t d
        v = v.unsqueeze(2).expand(-1, -1, t, -1, -1).flatten(-2) # B H t C*d
        
        x = self.qkv_attn(q, k, v) # B H t+1 C*d
        
        # Reshaping input to original size (mixing back tokens)
        x = x.view(B, H, t, C, d).permute((0, 3, 2, 1, 4)).flatten(-2) # B C t D
        x = x.view(B, N, D) # B C*t D

        # Projection back to embedding space
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
    
    def qkv_attn(self, q,k,v):
        '''
        Default attention mechanism
        '''
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return (attn @ v) # B H N d
    
    
class Attention(nn.Module):
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        '''
        Vanilla/Default attention
        '''
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False # use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            x = q @ k.transpose(-2, -1)
            x = x.softmax(dim=-1)
            x = self.attn_drop(x)
            x = x @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x) 
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class CustomAttentionBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.0,
            attn_drop: float = 0.0,
            init_values: Optional[float] = None,
            drop_path: float = 0.0,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            num_channels: int = 23,
            attention_type: str = 'default',
            block_idx = 0,
    ) -> None:
        super().__init__()
        '''
        Wrapper class for the 4 types of attentions available in the repository:
        1. Default attention: attention_type = 'default'
        2. TwoAxis attention: attention_type = 'twoaxis'
        3. Alternating attention: attention_type = 'alternating'
        4. Bottleneck attention: attention_type = 'bottleneck'
        
        WARNING: 
        Do not use any split attention mechanism for ViTMAE encoder.
        The machanism used to compute the input for the model does not preserve channel information, which will 
        therefore result in a bug in the reshaping of the tokens to the Channels, Patches shape.
        '''
        self.norm1 = norm_layer(dim)
        if attention_type == 'default':
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer
            )
        elif attention_type == 'twoaxis':          
            self.attn = TwoAxisAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                num_channels=num_channels
            )
        elif attention_type == 'alternating':
            self.attn = AlternatingAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
                block_idx=block_idx
            )
        elif attention_type == 'bottleneck':
            self.attn = BottleneckSplitAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                norm_layer=norm_layer,
            )
            
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    
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

from models.modules.pos_embed import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
from models.modules.patching import PatchingModule


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N = batch_size, L = num_tokens, D = embed_dim]
    """
    N, L, D = x.shape  # N = batch_size, L = num_tokens, D = embed_dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


class ChannelVisionTransformer(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=23,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, masking_ratio=0.75, keep_chans=True, using_spectrogram=True,
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
        
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        if using_spectrogram:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
            
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.blocks = nn.ModuleList([
            CustomAttentionBlock(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, attention_type=attention_type, block_idx=i, drop_path=drop_path)
            for i in range(depth)])
        self.norm = norm_layer(self.embed_dim)

        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.using_spectrogram:
            if self.keep_chans and self.square_patches:
                pos_embed = get_3d_sincos_pos_embed(embed_dim=self.embed_dim, grid_size=self.patches_per_dim, num_channels=self.in_chans, cls_token=True)
            elif self.keep_chans:
                pos_embed = get_2d_sincos_pos_embed(embed_dim=self.embed_dim, grid_size_height=1, grid_size_width=self.patch_embed.num_patches, cls_token=True)
            else:
                pos_embed = get_2d_sincos_pos_embed(embed_dim=self.embed_dim, grid_size_height=int(self.patch_embed.num_patches**.5), grid_size_width=int(self.patch_embed.num_patches**.5), cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        self.patch_embed.init_patch_embed()
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # Perform masking & prepend CLS token
    def prepare_tokens(self, x):
        B, _, _ = x.shape
        x, mask, ids_restore = random_masking(x, self.masking_ratio)
        # prepend cls token to the visible tokens
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # Add positional embeddings to all tokens except CLS token which already has its own
        x[:, 1:, :] = x[:, 1:, :] + self.pos_embed[:, 1:, :]
        return x, mask, ids_restore
    
    def forward(self, x, mask_tokens=True): #mask_tokens = True if doing MAE pre-training. False for fine-tuning.
        # Patch embedding
        x = self.patch_embed(x)  # Shape: [B, num_patches, embed_dim]
        
        if mask_tokens:
            x, mask, ids_restore = self.prepare_tokens(x)
        else:
            # For fine-tuning, just add positional embeddings without masking
            B = x.shape[0]
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x[:, 1:, :] = x[:, 1:, :] + self.pos_embed[:, 1:, :]
            
        x = x[:, 1:, :] # we remove the CLS token by default because we use Split Attention
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        if mask_tokens:
            return x, mask, ids_restore
        else:
            return x

    