import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed


def patchify_spectrogram(images, patch_size, keep_chans=True, square_patches=False):
    B, C, F, T = images.shape
    if square_patches:
        assert F % patch_size == 0 and T % patch_size == 0, "Image dimensions must be divisible by the patch size"
    else:
        assert T % patch_size == 0, "Time bins must be divisible by the patch size"
    p = patch_size
    t = f = F // p
    if keep_chans and square_patches:
        patches = images.reshape(B, C, f, patch_size, t, patch_size)
        patches = rearrange(patches, 'B C f p t q -> B (f t C) (p q)')
    elif keep_chans:
        patches = rearrange(images, 'b c f (t p) -> b (t c) (f p)', p=patch_size)
    else:
        patches = images.reshape(B, C, f, patch_size, t, patch_size)
        patches = rearrange(patches, 'B C f p t q -> B (f t) (p q C)')
    return patches

def patchify_waveform(images, patch_size, keep_chans):
    B, C, T = images.shape
    assert T % patch_size == 0 ,"Signal dimensions must be divisible by the patch size"
    p = patch_size
    t = T // p
    patches = images.reshape(B, C, t, patch_size)
    if keep_chans:
        patches = rearrange(patches, 'B C t p -> B (C t) p')
    else:
        patches = rearrange(patches, 'B C t p -> B t (p C)')
    return patches
    
def patchify(images, patch_size, keep_chans=True, using_spectrogram=True, square_patches=False):
    if using_spectrogram:
        return patchify_spectrogram(images, patch_size, keep_chans, square_patches)
    else:
        return patchify_waveform(images, patch_size, keep_chans)

def unpatchify_spectrogram(patches, patch_size, height, width, num_channels, keep_chans=True, square_patches=False):
    B, num_patches, _ = patches.shape
    F = height
    T = width
    C = num_channels
    f = F // patch_size
    t = T // patch_size
    p = patch_size
    if keep_chans:
        if square_patches:
            patches = patches.reshape(B, f, t, C, patch_size, patch_size)
            patches = rearrange(patches, 'B f t C p q -> B C (f p) (t q)')   
        else:
            patches = rearrange(patches, 'B (t C) (F p) -> B C F (t p)', C=C, F=F, p=patch_size, t=t)     
    else:
        f = t = int(num_patches**0.5)
        assert f * t == patches.shape[1]
        patches = patches.reshape(shape=(B, f, t, p, p, C))
        patches = rearrange(patches, 'B f t p q C -> B C (f p) (t q)')
    return patches

def unpatchify_waveform(patches, patch_size, length, num_channels, keep_chans=True):
    B, num_patches, _ = patches.shape
    T = length
    C = num_channels
    t = T // patch_size
    if keep_chans:
        patches = rearrange(patches, 'B (C t) p -> B C (t p)', C=C)        
    else:
        patches = rearrange(patches, 'B t (p C) -> B C (t p)', C=C)
    return patches

def unpatchify(patches, patch_size, num_channels, height=0, width=0, length=0, keep_chans=True, using_spectrogram=True, square_patches=False):
    if using_spectrogram:
        return unpatchify_spectrogram(patches, patch_size, height, width, num_channels, keep_chans, square_patches)
    else:
        return unpatchify_waveform(patches, patch_size, length, num_channels, keep_chans)
        
class PatchEmbedWaveform(nn.Module):
    """Image to Patch Embedding with Channel Embeddings."""
    def __init__(
        self,
        img_size: int = 1280,
        patch_size: int = 20,
        in_chans: int = 23,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = self.img_size // self.patch_size 
        self.proj = nn.Conv1d(
                in_chans,
                embed_dim,
                kernel_size=(self.patch_size),
                stride=(self.patch_size),
            )
        
        
    def forward(self, x):
        B, Cin, T = x.shape
        x = self.proj(x) 
        x = rearrange(x, 'B D t -> B t D')
        return x
    
class PatchEmbedPerChannel(nn.Module):
    """Image to Patch Embedding with Channel Embeddings."""
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 23,
        embed_dim: int = 1024,
        using_spectrogram = True
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.using_spectrogram = using_spectrogram
        if using_spectrogram:
            self.num_patches = (self.img_size // self.patch_size) * (self.img_size // self.patch_size) * self.in_chans
            self.proj = nn.Conv3d(
                1,
                embed_dim,
                kernel_size=(1, self.patch_size, self.patch_size),
                stride=(1, self.patch_size, self.patch_size),
            ) 
        else:
            self.num_patches = (self.img_size // self.patch_size) * self.in_chans
            self.proj = nn.Conv2d(
                1,
                embed_dim,
                kernel_size=(1, self.patch_size),
                stride=(1, self.patch_size),
            )

    def forward(self, x):
        if self.using_spectrogram:
            B, Cin, H, W = x.shape
            # shared projection layer across channels
            x = self.proj(x.unsqueeze(1)) 
            x = rearrange(x, 'B D C f t -> B (C f t) D')
        else:
            B, Cin, T = x.shape
            # shared projection layer across channels
            x = self.proj(x.unsqueeze(1)) 
            #x = rearrange(x, 'B D C t -> B (C t) D')
            B, D, C, t = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, -1, D)
        return x
    
class PatchEmbedLinear(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=23, embed_dim=768, square_patches=False):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size 
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.square_patches = square_patches

        if self.square_patches:
            self.num_patches = self.in_chans * ((self.img_size // self.patch_size) ** 2)
            self.proj = nn.Linear(self.patch_size ** 2, self.embed_dim)
        else:
            self.num_patches = self.in_chans * (self.img_size // self.patch_size)
            self.proj = nn.Linear(self.img_size * self.patch_size, self.embed_dim)

    def forward(self, x):
        patches = patchify(x, patch_size=self.patch_size, square_patches=self.square_patches)
        x = self.proj(patches)
        return x
    
    
class PatchingModule(nn.Module):
    """Image to Patch Embedding of choice according to the parameters given."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 23,
        embed_dim: int = 1024,
        keep_chans=True,
        using_spectrogram=False,
        square_patches=False
    ):
        super().__init__()
        '''
        Perform patching as desired based on the given parameters
        keep_chans = True/False : describes whether the channel dimension 
                                    should be preserved (keep_chans=True) 
                                    in the process of patchifying / tokenizing
        using_spectrogram = True/False : set to True if the expected input is a 
                                    Spectrogram (C Channels x 2d spectrogram), 
                                    set to False if the expected input is a
                                    Waveform (C Channels x 1d signal)
        square_patches = True/False : used only with using_spectrogram=True
                                    describes whether to use ViT-type square patching or
                                    to tokenize by binning along spectrogram time-steps
        '''
        self.using_spectrogram = using_spectrogram
        self.keep_chans = keep_chans
        self.square_patches = square_patches
     
        if self.using_spectrogram:
            if self.keep_chans:
                self.patch_embed = PatchEmbedLinear(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, square_patches=self.square_patches)
            else:
                self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        else:
            if self.keep_chans:
                self.patch_embed = PatchEmbedPerChannel(img_size, patch_size, in_chans, embed_dim, self.using_spectrogram)
            else:
                self.patch_embed = PatchEmbedWaveform(img_size, patch_size, in_chans, embed_dim, self.keep_chans)
                
        self.num_patches = self.patch_embed.num_patches
        self.init_patch_embed()
        
    def init_patch_embed(self):        
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        
    def forward(self, x):
        return self.patch_embed(x)
