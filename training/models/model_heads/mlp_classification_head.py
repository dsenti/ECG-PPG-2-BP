import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class FcBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(in_dim, out_dim)
        self.act_layer = None
        if act_layer:
            self.act_layer = act_layer()
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        if self.act_layer:
            x = self.act_layer(x)
        return x
        
class WeightedAverage(nn.Module):
    def __init__(self, nr_vectors, embed_dim):
        super().__init__()
        self.weights = nn.Linear(nr_vectors, 1, bias=False)

    def forward(self, x):
        weights = torch.softmax(self.weights.weight, dim=-1)
        x = torch.matmul(weights, x)
        return x.squeeze(1)


class MlpClassificationHead(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_layers,
        num_classes,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super(MlpClassificationHead, self).__init__()
                
        layers = []
        print(f'building model with embed_dim={embed_dim}, hidden_layers={hidden_layers}, num_classes={num_classes}')
        last_layer_inp = embed_dim
        if hidden_layers:
            print('hidden layer active')
            layers.append(FcBlock(embed_dim, hidden_layers[0], act_layer, drop))
            for i, layer_dim in enumerate(hidden_layers):
                if i == 0:
                    continue
                layers.append(FcBlock(hidden_layers[i-1], layer_dim, act_layer, drop))
            last_layer_inp = hidden_layers[-1]
        layers.append(FcBlock(last_layer_inp, num_classes, nn.GELU, drop))
        self.mlp = nn.Sequential(*layers)
        self.num_classes = num_classes
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
      
        
    def forward(self, x):
        #print(x)
        #if len(x.shape) == 3:  # (B, nr_vectors, embed_dim), compute weighted average along nr_vectors
        #if len(x) == 3:
        x = x.mean(dim=1)
        x = self.mlp(x)
        return x
    