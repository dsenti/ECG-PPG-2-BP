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
            num_channels = 23,
            block_idx   = 0
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.do_channel_attn = block_idx % 2 == 0
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        """
        attn_mask shape (B, N)
        """
        B, N, D = x.shape
        T = N // self.num_channels

        if self.do_channel_attn:
            # If the current block is a channel attention block, perform attention over the Channels dimension C
            #x = rearrange(x, 'B (C T) D -> (B T) C D', C=self.num_channels)
            B, _, D = x.shape
            C = self.num_channels
            T = x.shape[1] // C

            x = x.view(B, C, T, D).permute(0, 2, 1, 3).reshape(B * T, C, D)
            if attn_mask is not None:
                #attn_mask = rearrange(attn_mask, 'B (C T) -> (B T) C', C=self.num_channels)
                attn_mask = attn_mask.view(B, self.num_channels, T).permute(0, 2, 1).reshape(B * T, self.num_channels)
            x = self.forward_attn(x, attn_mask)
            #x = rearrange(x, '(B T) C D -> B (C T) D', T=T)
            x = x.view(B, T, C, D).permute(0, 2, 1, 3).reshape(B, C * T, D)
        else:
            # If the current block is a patch-attention block, perform attention over the patches dimension T
            #x = rearrange(x, 'B (C T) D -> (B C) T D', C=self.num_channels)
            x = x.view(B, self.num_channels, T, D).reshape(B * self.num_channels, T, D)

            if attn_mask is not None:
                #attn_mask = rearrange(attn_mask, 'B (C T) -> (B C) T', C=self.num_channels)
                attn_mask = attn_mask.view(B, self.num_channels, T).reshape(B * self.num_channels, T)
            x = self.forward_attn(x, attn_mask)
            #x = rearrange(x, '(B C) T D -> B (C T) D', C=self.num_channels)
            x = x.view(B, self.num_channels, T, D).reshape(B, self.num_channels * T, D)
        return x
    
    def forward_attn(self, x, attn_mask=None): 
        '''Default attention mechanism, assuming attn_mask has shape (B, N) and attn_mask == 0 : padded'''
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1).expand(B, self.num_heads, N, N).bool()
            
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask, 
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            
            if attn_mask is not None:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1).expand(B, self.num_heads, N, N)
                attn = attn.masked_fill(attn_mask == 0, float('-inf'))
            
            attn = attn.softmax(dim=-1)
            
            if attn_mask is not None:
                # Check for padded tensors and set them to zero after softmax
                fully_padded_idx = attn_mask.sum(dim=-1, keepdim=True).eq(0)
                attn = attn.masked_fill(fully_padded_idx, 0.0)
            
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
        
    def forward(self, x, attn_mask=None):
        
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

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
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
                num_channels=num_channels,
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

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    
class TwoAxisAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            num_channels: int = 23
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        self.qkv_channel = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_token = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.num_channels = num_channels


    def forward(self, x, attn_mask=None):
        B, N, D = x.shape
        tokens_per_channel = N // self.num_channels
        x = x.view(B, self.num_channels, tokens_per_channel, D)

        # Inter-channel attention
        x_channel = x.permute(0, 2, 1, 3)  # [B, tokens_per_channel, num_channels, D]
        x_channel = self._attention(x_channel.reshape(B*tokens_per_channel, self.num_channels, D), self.qkv_channel)
        x_channel = x_channel.view(B, tokens_per_channel, self.num_channels, D)
        x_channel = x_channel.permute(0, 2, 1, 3)  # [B, num_channels, tokens_per_channel, D]

        # Intra-channel attention
        x_token = self._attention(x.reshape(B*self.num_channels, tokens_per_channel, D), self.qkv_token)
        x_token = x_token.view(B, self.num_channels, tokens_per_channel, D)

        # Combine results
        x = x_channel + x_token
        x = x.reshape(B, N, D)

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _attention(self, x, qkv):
        B, N, C = x.shape
        qkv = qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x