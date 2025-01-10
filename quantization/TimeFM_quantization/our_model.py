import torch
import torch.nn as nn
import torch.nn.functional as F
from models import simMiM
import models.model_heads.mlp_classification_head as heads
import torch.ao.quantization as tq

class FinetunePretrainedModelSmall(nn.Module):
    def __init__(self, embed_dim=192, num_blocks=8, num_classes=2, drop_prob=0.2, quantize=False):
        super(FinetunePretrainedModelSmall, self).__init__()
        self.quantize = quantize
        if self.quantize:
            self.quant = tq.QuantStub()  # Quantization entry point
        self.model = simMiM.ChannelVisionTransformer(
            patch_size=64,
            embed_dim=192,
            num_heads=12,
            depth=8,
            in_chans=2,
            attention_type='alternating',
            using_spectrogram=False,
            drop_path=0.2,
            img_size=1250,
        )
        self.model_head = heads.MlpClassificationHead(
            embed_dim=192,
            hidden_layers=None, #[64]
            num_classes=2,
            act_layer=nn.GELU,
            drop=0.0,
        )
        if self.quantize:
            self.dequant = tq.DeQuantStub()  # Dequantization entry point

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)  # Quantize input
        x = self.model(x, mask_tokens=False)
        #x = F.gelu(x)
        x = self.model_head(x)
        if self.quantize:
            x = self.dequant(x)  # Dequantize output
        return x

class FinetunePretrainedModelBase(nn.Module):
    def __init__(self, embed_dim=192, num_blocks=8, num_classes=2, drop_prob=0.2, quantize=False):
        super(FinetunePretrainedModelBase, self).__init__()
        self.quantize = quantize
        if self.quantize:
            self.quant = tq.QuantStub()  # Quantization entry point
        self.model = simMiM.ChannelVisionTransformer(
            patch_size=64,
            embed_dim=576,
            num_heads=12,
            depth=10,
            in_chans=2,
            attention_type='alternating',
            using_spectrogram=False,
            drop_path=0.2,
            img_size=1250,
        )
        self.model_head = heads.MlpClassificationHead(
            embed_dim=576,
            hidden_layers=None, #[64]
            num_classes=2,
            act_layer=nn.GELU,
            drop=0.0,
        )
        if self.quantize:
            self.dequant = tq.DeQuantStub()  # Dequantization entry point

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)  # Quantize input
        x = self.model(x, mask_tokens=False)
        #x = F.gelu(x)
        x = self.model_head(x)
        if self.quantize:
            x = self.dequant(x)  # Dequantize output
        return x

class FinetunePretrainedModelLarge(nn.Module):
    def __init__(self, embed_dim=192, num_blocks=8, num_classes=2, drop_prob=0.2, quantize=False):
        super(FinetunePretrainedModelLarge, self).__init__()
        self.quantize = quantize
        if self.quantize:
            self.quant = tq.QuantStub()  # Quantization entry point
        self.model = simMiM.ChannelVisionTransformer(
            patch_size=64,
            embed_dim=768,
            num_heads=12,
            depth=12,
            in_chans=2,
            attention_type='alternating',
            using_spectrogram=False,
            drop_path=0.2,
            img_size=1250,
        )
        self.model_head = heads.MlpClassificationHead(
            embed_dim=768,
            hidden_layers=None, #[64]
            num_classes=2,
            act_layer=nn.GELU,
            drop=0.0,
        )
        if self.quantize:
            self.dequant = tq.DeQuantStub()  # Dequantization entry point

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)  # Quantize input
        x = self.model(x, mask_tokens=False)
        #x = F.gelu(x)
        x = self.model_head(x)
        if self.quantize:
            x = self.dequant(x)  # Dequantize output
        return x
