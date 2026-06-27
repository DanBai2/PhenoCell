"""
Model unit from OpenAI CLIP [1]
[1] https://github.com/openai/CLIP/blob/main/clip/model.py
[2] https://arxiv.org/abs/2503.10622
[3] https://huggingface.co/google-bert/bert-base-cased
"""
import os
from collections import OrderedDict
from functools import partial
from typing import List, Tuple, Union
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPImageProcessor,
    CLIPVisionModel,
    SiglipImageProcessor,
    SiglipVisionModel,
)
import numpy as np
import torch
import torch.nn.functional as F
# from graphium.config._loader import (
#     load_accelerator,
#     load_architecture,
#     load_datamodule,
#     load_yaml_config,
# )
from torch import nn
from torch_geometric.data import Batch, Data
from transformers import BertModel
from typing import Union, List, Dict, Optional
from transformers import AutoTokenizer, AutoModel, BertModel
from src import constants
from src.channelvit.backbone.channel_vit import ChannelVisionTransformer as ChannelViT
from src.channelvit.backbone.hcs_channel_vit import (
    ChannelVisionTransformer as ChannelViT_hcs,
)
from src.channelvit.utils.optim import trunc_normal_
# from src.mpnn.model import FullGraphMultiTaskNetworkNew
# from src.open_phenom.hugginface_mae import MAEModel
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPImageProcessor,
    CLIPVisionModel,
    SiglipImageProcessor,
    SiglipVisionModel,
)

class Bottleneck(nn.Module):
    """Bottleneck layer"""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()

        # all conv layers have stride 1.
        # an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool,
            # and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    """Attention"""

    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5
        )

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):

        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class AttentionPooling(nn.Module):
    """Atteneion pooling for cell embeddings."""

    def __init__(self, dim, reduction_factor: int = 2):
        super(AttentionPooling, self).__init__()

        hidden_dim = max(dim // reduction_factor, 128)
        self.query_fc = nn.Linear(dim, hidden_dim)
        self.key_fc = nn.Linear(dim, hidden_dim)
        self.value_fc = nn.Linear(dim, dim)

    def forward(self, cell_embeddings):
        """Forward for attention"""

        queries = self.query_fc(cell_embeddings)
        keys = self.key_fc(cell_embeddings)
        values = self.value_fc(cell_embeddings)

        attention_scores = torch.matmul(queries, keys.T) / (keys.shape[-1] ** 0.5)
        attention_weights = F.softmax(attention_scores.mean(dim=0), dim=0)

        pooled_embedding = torch.sum(values * attention_weights.unsqueeze(1), dim=0)

        return pooled_embedding.unsqueeze(0)  # Shape: (1, dim)


class MILPooling(nn.Module):
    """Channel-independent pooling with gated attention for multi-instance learning"""

    def __init__(self, input_dim, hidden_dim=128, pooling="mean"):
        super().__init__()

        self.pooling = pooling

        if self.pooling == "attention":
            self.V = nn.Linear(input_dim, hidden_dim)
            self.U = nn.Linear(input_dim, hidden_dim)
            self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, M, C, D)
        Return:
        ------
            bag_rep: Tensor of shape (B, C, D)
        """
        B, M, C, D = x.shape

        if self.pooling == "attention":

            x = x.permute(0, 2, 1, 3)  # (B, C, M, D)
            x = x.reshape(B * C, M, D)  # (B*C, M, D)

            h_V = torch.tanh(self.V(x))  # (B*C, M, hidden_dim)
            h_U = torch.sigmoid(self.U(x))  # (B*C, M, hidden_dim)

            h = h_V * h_U  # (B*C, M, hidden_dim)

            attn_scores = self.attention(h)  # (B*C, M, 1)

            mask = (x.abs().sum(dim=-1) > 0).float()  # (B*C, M)
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))

            attn_weights = torch.softmax(attn_scores, dim=1)  # (B*C, M, 1)

            bag_rep = torch.sum(attn_weights * x, dim=1)  # (B*C, D)
            bag_rep = bag_rep.view(B, C, D)  # (B, C, D)
        elif self.pooling == "mean":
            bag_rep = torch.mean(x, axis=1)
        elif self.pooling == "median":
            bag_rep = torch.median(x, axis=1)[0]

        return bag_rep


class MILChannelIndependentPooling(nn.Module):
    """Channel-independent pooling with gated attention for multi-instance learning"""

    def __init__(self, input_dim, hidden_dim=128, num_channels=5):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        # Combine all channels into a single weight tensor
        self.V_weight = nn.Parameter(torch.randn(num_channels, hidden_dim, input_dim))
        self.V_bias = nn.Parameter(torch.zeros(num_channels, hidden_dim))
        self.U_weight = nn.Parameter(torch.randn(num_channels, hidden_dim, input_dim))
        self.U_bias = nn.Parameter(torch.zeros(num_channels, hidden_dim))
        self.attn_weight = nn.Parameter(
            torch.randn(num_channels, hidden_dim, 1).contiguous()
        )
        self.attn_bias = nn.Parameter(torch.zeros(num_channels, 1))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, M, C, D)
        Return:
        ------
            bag_rep: Tensor of shape (B, C, D)
        """
        B, M, C, D = x.shape
        assert C == self.num_channels

        # Rearrange to (C, B, M, D)
        x = x.permute(2, 0, 1, 3)  # (C, B, M, D)

        # Apply V and U linears via batch matmul
        x_ = x.reshape(C, B * M, D)

        # Project using gated attention
        h_V = torch.tanh(
            torch.bmm(x_, self.V_weight.transpose(1, 2)) + self.V_bias.unsqueeze(1)
        )  # (C, B*M, H)
        h_U = torch.sigmoid(
            torch.bmm(x_, self.U_weight.transpose(1, 2)) + self.U_bias.unsqueeze(1)
        )  # (C, B*M, H)
        h = h_V * h_U  # (C, B*M, H)

        # Compute attention scores
        attn_logits = torch.bmm(h, self.attn_weight) + self.attn_bias.unsqueeze(
            1
        )  # (C, B*M, 1)
        attn_logits = attn_logits.view(C, B, M, 1)  # (C, B, M, 1)

        # Mask padding
        mask = (x.abs().sum(dim=-1) > 0).unsqueeze(-1)  # (C, B, M, 1)
        attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

        attn_weights = torch.softmax(attn_logits, dim=2)  # (C, B, M, 1)
        pooled = torch.sum(attn_weights * x, dim=2)  # (C, B, D)
        pooled = pooled.permute(1, 0, 2)  # (B, C, D)

        return pooled.contiguous()


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1,
        with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions,
        where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(
        self,
        layers,
        output_dim,
        heads,
        input_resolution=224,
        width=64,
        input_channels=3,
        num_classes=10,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            input_channels, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(
            input_resolution // 32, embed_dim, heads, output_dim
        )

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class DynamicTanh(nn.Module):
    """DynamicTanh layer to replace layernorm from [2]"""

    def __init__(self, normalized_shape, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return self.weight * torch.tanh(self.alpha * x) + self.bias


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """QGELU"""

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """ResidualAttentionBlock"""

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Transformer backbone"""

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CrossChannelFormer(nn.Module):
    """Transformer backbone that takes multi-channel CLIP embeddings as input."""

    def __init__(
        self,
        embed_dim: int,
        layers: int,
        heads: int,
        output_dim: int,
        input_channels: int = 3,
        use_cls_token: bool = True,
        # use_channel_emb = True
    ):
        super().__init__()

        scale = embed_dim ** -0.5

        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.input_channels = input_channels
        self.use_cls_token = use_cls_token
        # self.use_channel_emb = use_channel_emb

        # if self.use_channel_emb:
        self.channel_embed = nn.Parameter(torch.zeros(input_channels, embed_dim))
        trunc_normal_(self.channel_embed, std=0.02)

        self.channel_ln = LayerNorm(embed_dim)

        self.ln_pre = LayerNorm(embed_dim)

        self.transformer = Transformer(embed_dim, layers, heads)
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.ln_post = LayerNorm(embed_dim)
        self.proj = nn.Parameter(scale * torch.randn(embed_dim, output_dim))

    def forward(self, x: torch.Tensor):

        # print(f"image emb input shape is{x.shape}")

        B, _, _ = x.shape

        # add channel embeddings.

        # if self.use_channel_emb:
        channel_emb = self.channel_embed.expand(B, -1, -1)
        x += channel_emb.to(x.dtype)

        x = self.channel_ln(x)

        # add the [CLS] token to the embed patch tokens

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.use_cls_token:
            x = self.ln_post(x[:, 0, :])
        else:
            x = self.ln_post(x[:, 1:, :].mean(dim=1))

        if self.embed_dim != self.output_dim:
            x = x @ self.proj

        return x


class VisionTransformer(nn.Module):
    """Vision transformre"""

    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        input_channels: int = 3,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CrossChannelClassifier(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        layers: int,
        heads: int,
        output_dim: int,
        num_classes: int,
        input_channels: int = 3,
        use_cls_token: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        

        self.backbone = CrossChannelFormer(
            embed_dim=embed_dim,
            layers=layers,
            heads=heads,
            output_dim=output_dim,
            input_channels=input_channels,
            use_cls_token=use_cls_token
        )
        

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        return self.classifier(features)


class ResNet(nn.Module):
    """Image encoder used in cloome."""

    def __init__(
        self,
        block="bottleneck",
        layers: list = (3, 4, 23, 3),
        input_shape=None,
        output_dim=None,
        regression=False,
    ):
        self.inplanes = 64
        self.input_resolution = input_shape

        super().__init__()

        if block == "bottleneck":
            block = Bottleneck
        # elif block == "basic":
        #     block = BasicBlock
        # self.n_classes = num_classes
        if input_shape is not None:
            channels_in = input_shape
        else:
            channels_in = 3

        self.is_regression = regression
        self.conv1 = nn.Conv2d(
            channels_in, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if x.shape[-2:] != (1, 1):
            x = nn.AvgPool2d(x.shape[2:])(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class MLP(nn.Module):
    """Chemical encoder for cloome."""

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()

        self.layers = nn.ModuleList()

        for layer in range(n_layers):
            dim = input_dim if layer == 0 else hidden_dim
            self.layers.append(
                nn.Sequential(
                    nn.Linear(dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
                )
            )

        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, output_dim)))
        self.init_weights()

    def init_weights(self):
        """
        Initializes weights using He initialization for Linear layers and biases to zero.
        BatchNorm layers are initialized with default settings for weights and biases.
        """

        for idx, layer in enumerate(self.layers):
            if isinstance(layer, nn.Sequential):
                for sublayer in layer:
                    if isinstance(sublayer, nn.Linear):
                        if idx == len(self.layers) - 1:
                            nn.init.kaiming_normal_(
                                sublayer.weight, mode="fan_out", nonlinearity="linear"
                            )
                            nn.init.constant_(sublayer.bias, 0)
                        else:
                            nn.init.kaiming_normal_(
                                sublayer.weight, mode="fan_out", nonlinearity="relu"
                            )
                            nn.init.constant_(sublayer.bias, 0.1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CLIP(nn.Module):
    """CLIP initialization"""

    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        input_channels: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        # params for long clip
        long_clip: bool = False,
    ):
        super().__init__()

        self.context_length = 248

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )
        else:
            vision_heads = vision_width // 64

            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                input_channels=input_channels,
            )
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.long_clip = long_clip

        if long_clip:
            self.positional_embedding = nn.Parameter(torch.empty(248, transformer_width))
            self.positional_embedding_res = nn.Parameter(
                torch.empty(248, transformer_width)
            )
        else:
            self.positional_embedding = nn.Parameter(torch.empty(77, transformer_width))

        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self.mask1 = torch.zeros([248, 1])
        self.mask1[:20, :] = 1
        self.mask2 = torch.zeros([248, 1])
        self.mask2[20:, :] = 1

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                self.visual.layer1,
                self.visual.layer2,
                self.visual.layer3,
                self.visual.layer4,
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )

        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask,
        # with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        # if self.long_clip:
        x = (
            x
            + (self.positional_embedding * self.mask1.to(x.device)).type(self.dtype)
            + (self.positional_embedding_res * self.mask2.to(x.device)).type(self.dtype)
        )
        # else:
        #     x = x + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding
        # (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return image_features, text_features, self.logit_scale


class BERT_CLIP(nn.Module):
    """CLIP with bert-base-cased [3] as text encoder"""

    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: List[int],
        vision_width: int,
        input_channels: int,
        # text
        context_length: int,
    ):
        super().__init__()

        self.context_length = context_length
        self.visual = ResNet(
            layers=vision_layers, output_dim=embed_dim, input_shape=input_channels
        )
        # load pre-trained text-encoder
        self.text = BertModel.from_pretrained("bert-base-cased")
        self.text_proj = nn.Linear(768, 512)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                self.visual.layer1,
                self.visual.layer2,
                self.visual.layer3,
                self.visual.layer4,
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        out = self.text(
            input_ids=text["input_ids"], attention_mask=text["attention_mask"]
        ).pooler_output.type(self.dtype)
        out = self.text_proj(out)

        return out

    def forward(self, image, text):

        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features, self.logit_scale


class Molphenix(nn.Module):
    """Molphenix with MAE (Openphenom-S) and MPNN++"""

    def __init__(self, embed_dim, vision_width, vision_heads):
        super().__init__()
        # Load pre-trained MAE from huggingface with hiddend dim = 384
        self.visual = Transformer(vision_width, 6, vision_heads)
        self.mol_encoder = Transformer(vision_width, 1, vision_heads)

        self.phenom_proj = nn.Linear(vision_width, embed_dim)
        self.mol_proj = nn.Linear(vision_width, embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.1))
        self.bias = nn.Parameter(torch.ones([]) * -1)

    @property
    def dtype(self):
        return self.visual.resblocks[0].mlp.c_fc.weight.dtype

    def encode_image(self, image):
        image = self.visual(image).type(self.dtype)
        image = self.phenom_proj(image)
        return image

    def encode_mols(self, mols):
        mols = self.mol_encoder(mols).type(self.dtype)
        mols = self.mol_proj(mols)
        return mols

    def forward(self, image, mols):

        image_features = self.encode_image(image)
        mol_features = self.encode_mols(mols)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        mol_features = mol_features / mol_features.norm(dim=-1, keepdim=True)

        return image_features, mol_features, self.logit_scale, self.bias


class CellCLIP_MAE(nn.Module):
    """CLIP with Biomedical CLIP as text encoder and OPenpnenom as image encoder"""

    def __init__(
        self,
        embed_dim: int,
        # text
        context_length: int,
        pretrained: bool,
    ):
        super().__init__()

        self.context_length = context_length

        # Load pre-trained MAE from huggingface
        self.input_norm = MAEModel.from_pretrained("recursionpharma/OpenPhenom").input_norm
        self.visual = MAEModel.from_pretrained(
            "recursionpharma/OpenPhenom", load_weights=pretrained
        ).encoder

        # load pre-trained text-encoder
        # model, _ = create_model_from_pretrained(
        #     "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        # )
        self.text = BertModel.from_pretrained("bert-base-cased")
        self.text_proj = nn.Linear(768, 384)

        # self.text = model.text
        # self.text_proj = nn.Linear(512, 384)

        # self.logit_scale = model.logit_scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        return self.visual.vit_backbone.blocks[0].mlp_out_proj.weight.dtype

    def encode_image(self, image):
        # use mean pooled features over non-CLS token

        image = self.input_norm(image)
        X = self.visual.vit_backbone.forward_features(
            image
        )  # 3d tensor N x num_tokens x dim
        X = self.visual.vit_backbone.fc_norm(X)

        latent = X[:, 1:, :].mean(dim=1)  # 1 + 256 * C tokens

        return latent.type(self.dtype)  # self.visual(image.type(self.dtype))

    # def encode_text(self, text):

    #     text_output = self.text(text).type(self.dtype)
    #     text_output = self.text_proj(text_output)

    #     return text_output
    def encode_text(self, text):
        out = self.text(
            input_ids=text["input_ids"], attention_mask=text["attention_mask"]
        ).pooler_output.type(self.dtype)
        out = self.text_proj(out)

        return out

    def forward(self, image, text):

        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features, self.logit_scale



class CellCLIP(nn.Module):
    """Multi-instance CellCLIP with control"""

    def __init__(
        self,
        embed_dim: int,
        vision_layers: List[int],
        vision_width: int,
        vision_heads: int,
        input_channels: int,
        context_length: int,
        pooling: str,
        use_bias: bool = False,
        text_model_path: str = "google-bert/bert-base-cased",
    ):
        super().__init__()

        self.context_length = context_length
        self.use_bias = use_bias

        self.visual = CrossChannelFormer(
            embed_dim=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            input_channels=input_channels,
        )

        self.image_pool = MILPooling(input_dim=vision_width, pooling=pooling)
        self.text = BertModel.from_pretrained(text_model_path)
        self.text_proj = nn.Linear(512, 512)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if self.use_bias:
            self.bias = nn.Parameter(torch.ones([]) * -10)


        self.control_proj = nn.Linear(embed_dim, embed_dim)
        

        self.fusion_proj = nn.Linear(512 * 2, 512)

    @property
    def dtype(self):
        return self.visual.transformer.resblocks[0].mlp.c_fc.weight.dtype

    def encode_mil(self, image):
        bag_feats = self.image_pool(image)  # (B, C, D)

        return bag_feats

    def encode_image(self, image):
        # print(f"image shape is {image.shape}")
        # image=self.image_pool(image)
        image_feats = self.visual(image).type(self.dtype)  # (B, embed_dim)

        return image_feats

    def encode_text(self, text):
        out = self.text(
            input_ids=text["input_ids"], attention_mask=text["attention_mask"]
        ).pooler_output.type(self.dtype)
        out = self.text_proj(out)

        return out

    def forward(self, image, text,control=None):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)


        if control is None:
            return image_features, text_features, self.logit_scale



class CrossModalAttention(nn.Module):
    """"""
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key, value):
        # query, key, value: (batch_size, 1, embed_dim)
        attn_output, attn_weights = self.multihead_attn(query, key, value)

        output = self.layer_norm(query + attn_output)
        return output

class CrossModalFusionBlock(nn.Module):
    """： +  + """
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        

        self.self_attention_text = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.self_attention_control = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.self_attention_norm_text = nn.LayerNorm(embed_dim)
        self.self_attention_norm_control = nn.LayerNorm(embed_dim)
        

        self.cross_attention_text_to_control = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self.cross_attention_control_to_text = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )
        self.cross_attention_norm_text = nn.LayerNorm(embed_dim)
        self.cross_attention_norm_control = nn.LayerNorm(embed_dim)
        

        self.feed_forward_text = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.feed_forward_control = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ff_norm_text = nn.LayerNorm(embed_dim)
        self.ff_norm_control = nn.LayerNorm(embed_dim)
        

        self.dropout = nn.Dropout(0.1)
        

        self._init_weights()
    
    def _init_weights(self):
        """InitializeWeights"""
        for module in [self.feed_forward_text, self.feed_forward_control]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, text_features, control_features):
        """
        Args:
            text_features: (batch_size, embed_dim)
            control_features: (batch_size, embed_dim)
        Returns:
            fused_text_features: (batch_size, embed_dim)
            fused_control_features: (batch_size, embed_dim)
        """

        text_seq = text_features.unsqueeze(1)  # (batch, 1, embed_dim)
        control_seq = control_features.unsqueeze(1)  # (batch, 1, embed_dim)
        

        text_self_attn, _ = self.self_attention_text(
            text_seq, text_seq, text_seq
        )
        text_features = self.self_attention_norm_text(
            text_features + text_self_attn.squeeze(1)
        )
        
        control_self_attn, _ = self.self_attention_control(
            control_seq, control_seq, control_seq
        )
        control_features = self.self_attention_norm_control(
            control_features + control_self_attn.squeeze(1)
        )
        

        text_seq = text_features.unsqueeze(1)
        control_seq = control_features.unsqueeze(1)
        


        text_cross_attn = self.cross_attention_text_to_control(
            query=text_seq,
            key=control_seq,
            value=control_seq
        ).squeeze(1)
        text_features = self.cross_attention_norm_text(
            text_features + text_cross_attn
        )
        

        control_cross_attn = self.cross_attention_control_to_text(
            query=control_seq,
            key=text_seq,
            value=text_seq
        ).squeeze(1)
        control_features = self.cross_attention_norm_control(
            control_features + control_cross_attn
        )
        

        text_ff = self.feed_forward_text(text_features)
        text_features = self.ff_norm_text(text_features + text_ff)
        
        control_ff = self.feed_forward_control(control_features)
        control_features = self.ff_norm_control(control_features + control_ff)
        

        text_features = self.dropout(text_features)
        control_features = self.dropout(control_features)
        
        return text_features, control_features



class TextEncoder(nn.Module):
    """TextEncoder (BioBERT)"""
    
    def __init__(
        self, 
        model_path: str = "google-bert/bert-base-cased",
        proj_dim: int = 512,
        freeze_backbone: bool = False
    ):
        super().__init__()
        

        self.model = BertModel.from_pretrained(model_path)
        self.hidden_size = self.model.config.hidden_size
        

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        

        self.proj = nn.Linear(self.hidden_size, proj_dim)
        self.proj_dim = proj_dim
    
    def forward(self, text: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            text:  input_ids  attention_mask 
        Returns:
            (B, proj_dim) TextEmbedding
        """

        outputs = self.model(
            input_ids=text["input_ids"],
            attention_mask=text["attention_mask"]
        )
        

        embeddings = outputs.pooler_output
        

        embeddings = self.proj(embeddings)
        
        return embeddings


class SMILESEncoder(nn.Module):
    """SMILESEncoder"""
    
    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        proj_dim: int = 512,
        max_length: int = 128,
        freeze_backbone: bool = True
    ):
        super().__init__()
        

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        self.max_length = max_length
        

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        

        self.proj = nn.Linear(self.hidden_size, proj_dim)
        self.proj_dim = proj_dim
    
    def forward(self, smiles: List[str]) -> torch.Tensor:
        """
        Args:
            smiles: SMILESList
        Returns:
            (B, proj_dim) CompoundEmbedding
        """
        # Tokenize
        inputs = self.tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(next(self.model.parameters()).device)
        

        with torch.set_grad_enabled(not all(p.requires_grad is False for p in self.model.parameters())):
            outputs = self.model(**inputs)
        

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        

        embeddings = self.proj(cls_embeddings)
        
        return embeddings
    
    def tokenize(self, smiles: List[str]) -> Dict[str, torch.Tensor]:
        """，Dataset"""
        return self.tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )


class MorganFingerprintEncoder(nn.Module):
    """Encoder (MLP)"""
    
    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 768,
        proj_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = nn.ModuleList()
        

        layers.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ))
        

        for _ in range(n_layers - 2):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        

        layers.append(nn.Linear(hidden_dim, proj_dim))
        
        self.layers = nn.Sequential(*layers)
        self.proj_dim = proj_dim
    
    def forward(self, fingerprints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fingerprints: (B, input_dim) 
        Returns:
            (B, proj_dim) Embedding
        """
        return self.layers(fingerprints)


class CompoundEncoder(nn.Module):
    """CompoundEncoder"""
    
    def __init__(
        self,
        mode: str = "text",  # "text", "smiles", "morgan"
        proj_dim: int = 512,

        text_model_path: str = "google-bert/bert-base-cased",
        freeze_text: bool = False,

        smiles_model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        max_smiles_length: int = 128,
        freeze_smiles: bool = True,

        morgan_dim: int = 1024,
        morgan_hidden_dim: int = 512,
        morgan_n_layers: int = 4
    ):
        super().__init__()
        
        self.mode = mode
        self.proj_dim = proj_dim
        

        if mode == "text":
            self.encoder = TextEncoder(
                model_path=text_model_path,
                proj_dim=proj_dim,
                freeze_backbone=freeze_text
            )
            
        elif mode == "smiles":
            self.encoder = SMILESEncoder(
                model_name=smiles_model_name,
                proj_dim=proj_dim,
                max_length=max_smiles_length,
                freeze_backbone=freeze_smiles
            )
            
        elif mode == "morgan":
            self.encoder = MorganFingerprintEncoder(
                input_dim=morgan_dim,
                hidden_dim=morgan_hidden_dim,
                proj_dim=proj_dim,
                n_layers=morgan_n_layers
            )
            
        elif mode == "plate":

            self.encoder = nn.Embedding(1000, proj_dim)
            
        elif mode == "embedding":

            self.encoder = nn.Identity()
            
        elif mode == "label":

            self.encoder = nn.Embedding(100, proj_dim)
            
        else:
            raise ValueError(f": {mode}")
    
    def forward(self, inputs: Union[Dict, List[str], torch.Tensor, str]) -> torch.Tensor:
        """
        
        Args:
            inputs: :
                - text: {"input_ids": tensor, "attention_mask": tensor}
                - smiles: List[str] (SMILESList)
                - : torch.Tensor
        Returns:
            (B, proj_dim) CompoundEmbedding
        """


        return self.encoder(inputs)
    
    def get_tokenizer(self):
        """tokenizer（text/smiles）"""
        if hasattr(self.encoder, 'tokenizer'):
            return self.encoder.tokenizer
        elif hasattr(self.encoder, 'model') and hasattr(self.encoder.model, 'tokenizer'):
            return self.encoder.model.tokenizer
        return None
    
    @property
    def output_dim(self) -> int:
        """"""
        return self.proj_dim




class PhenoCell(nn.Module):
    """Multi-instance CellCLIP with Cross-modal Attention"""

    def __init__(
        self,
        embed_dim: int,
        vision_layers: List[int],
        vision_width: int,
        vision_heads: int,
        input_channels: int,
        context_length: int,
        pooling: str,
        use_bias: bool = False,
        num_cross_modal_layers: int = 1,
        cross_modal_heads: int = 8,

        model_card: str = "facebook/dinov2-giant",
        text_model_path:str='google-bert/bert-base-cased',
        aggregation: str = None,
        max_instances: int = 6,
        mole_struc:str=None
    ):
        super().__init__()
        
        self.context_length = context_length
        self.use_bias = use_bias
        self.embed_dim = embed_dim
        self.num_cross_modal_layers = num_cross_modal_layers
        self.model_card = model_card
        self.aggregation = aggregation
        self.max_instances = max_instances

        # self.pretrained_clip=self.load_model()
        # self.aggregation = "mean"

        self.visual = CrossChannelFormer(
            embed_dim=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            input_channels=input_channels,
        )
        
        self.image_pool = MILPooling(input_dim=vision_width, pooling=pooling)
        

        self.text = BertModel.from_pretrained(
            "google-bert/bert-base-cased"
        )
        self.text_proj = nn.Linear(768, embed_dim)

        self.compound=CompoundEncoder(mode=mole_struc,proj_dim=embed_dim,text_model_path=text_model_path)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.ones([]) * -10)
        

        self.control_proj = nn.Linear(embed_dim, embed_dim)
        

        self.cross_modal_fusion_blocks = nn.ModuleList([
            CrossModalFusionBlock(
                embed_dim=embed_dim,
                num_heads=cross_modal_heads
            ) for _ in range(num_cross_modal_layers)
        ])
        

        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        

        self.attention_weights = None

    @property
    def dtype(self):
        return self.visual.transformer.resblocks[0].mlp.c_fc.weight.dtype
    
    def load_model(self):
        if "siglip" in self.model_card:
            pretrained_model = SiglipVisionModel.from_pretrained(self.model_card)
            processor = SiglipImageProcessor.from_pretrained(self.model_card)
            crop_size = processor.size["height"]
        elif "dino" in self.model_card:
            pretrained_model = AutoModel.from_pretrained(self.model_card)
            processor = AutoImageProcessor.from_pretrained(self.model_card)

            if self.model_card == "facebook/dino-vitb8":
                crop_size = processor.size["height"]
            else:
                crop_size = processor.crop_size["height"]

        elif "clip" in self.model_card:
            pretrained_model = CLIPVisionModel.from_pretrained(self.model_card)
            processor = CLIPImageProcessor.from_pretrained(self.model_card)
            crop_size = processor.crop_size["height"]
        elif self.model_card== "openphenom":
            pretrained_model = MAEModel.from_pretrained("recursionpharma/OpenPhenom")
            crop_size = 256
            processor = None

        return pretrained_model
    
    def generate_embeddings(self, batch_images):
        """
        GenerateImageEmbedding
        Args:
            batch_images: List[List[Tensor]] - Image
                List: batch_size
                List:  samplesImageList
                Image: (5, 3, H, W)
        Returns:
            Tensor: GenerateEmbedding，Shape
        """

        all_crops = []
        slide_lengths = []
        
        # for image_slides in batch_images:

        #     if len(image_slides) > 0:

        #         crops_tensor = torch.stack(image_slides)
        #         all_crops.append(crops_tensor)
        #         slide_lengths.append(crops_tensor.shape[0])
        #     else:

        #         all_crops.append(torch.zeros(0, 5, 3, *image_slides[0].shape[2:]))
        #         slide_lengths.append(0)
        

        # if len(all_crops) == 0 or all([c.shape[0] == 0 for c in all_crops]):

        #     if self.aggregation is None:
        #         return torch.zeros(len(batch_images), self.max_instances, self.embed_dim)
        #     else:
        #         return torch.zeros(len(batch_images), self.embed_dim)
        # print(f"generate_embeddings received: type={type(batch_images)}, len={len(batch_images)}")
        
        for i, image_slides in enumerate(batch_images):
            # print(f"  Processing sample {i}: type={type(image_slides)}")
            
            if torch.is_tensor(image_slides):
                # print(f"    Tensor shape: {image_slides.shape}")

                if image_slides.dim() == 5:  # (num_crops, 5, 3, H, W)
                    crops_tensor = image_slides
                elif image_slides.dim() == 4:
                    crops_tensor = image_slides.unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected tensor dimension: {image_slides.dim()}")
                
                all_crops.append(crops_tensor)
                slide_lengths.append(crops_tensor.shape[0])
            elif isinstance(image_slides, list):
                # print(f"    List length: {len(image_slides)}")
                if len(image_slides) > 0:

                    print(f"    First element in list type: {type(image_slides[0])}")
                    if torch.is_tensor(image_slides[0]):
                        print(f"    First element shape: {image_slides[0].shape}")
                    

                    crops_tensor = torch.stack(image_slides)
                else:


                    crops_tensor = torch.zeros(0, 5, 3, 224, 224)

                all_crops.append(crops_tensor)
                slide_lengths.append(crops_tensor.shape[0])
            else:
                raise TypeError(f"Unsupported type: {type(image_slides)}")
        
        

        all_crops_tensor = torch.cat(all_crops, dim=0)  # (total_crops, 5, 3, H, W)
        

        b, c, *img_shape = all_crops_tensor.shape
        flattened_tensor = all_crops_tensor.view(b * c, *img_shape)
        

        all_embeddings = []
        batch_size = 128
        
        with torch.no_grad():
            for batch_start in range(0, flattened_tensor.size(0), batch_size):
                batch_end = min(batch_start + batch_size, flattened_tensor.size(0))
                batch = flattened_tensor[batch_start:batch_end]
                if self.model_card == "openphenom":
                    embeddings = self.pretrained_clip.predict(batch)
                else:
                    embeddings = self.pretrained_clip(batch).pooler_output

                # embeddings = self.visual(batch)  # (batch_size, embed_dim)
                all_embeddings.append(embeddings)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)  # (total_crops * 5, embed_dim)
        

        final_embeddings_list = []
        start = 0
        
        for num_crops in slide_lengths:
            num_crops_total = num_crops * 5
            
            if num_crops_total > 0:
                slide_embeddings = all_embeddings[start : start + num_crops_total]
                start += num_crops_total
                

                slide_embeddings = slide_embeddings.view(num_crops, 5, -1)
                

                if self.aggregation == "mean":

                    final_embeddings = torch.mean(slide_embeddings, dim=0)  # (5, embed_dim)
                    final_embeddings = torch.mean(final_embeddings, dim=0)  # (embed_dim)
                    
                elif self.aggregation == "median":

                    median_embeddings = torch.median(slide_embeddings, dim=0).values  # (5, embed_dim)
                    final_embeddings = torch.mean(median_embeddings, dim=0)  # (embed_dim)
                    
                elif self.aggregation == "hierarchical_pooling":

                    final_embeddings = self._hierarchical_pooling(
                        slide_embeddings.cpu().numpy(), num_clusters=3, weighting="uniform"
                    )
                    final_embeddings = torch.from_numpy(final_embeddings).to(slide_embeddings.device)
                    
                elif self.aggregation == "gmm":

                    final_embeddings = self._gmm_pooling(
                        slide_embeddings.cpu().numpy(), num_clusters=3
                    )
                    final_embeddings = torch.from_numpy(final_embeddings).to(slide_embeddings.device)
                    
                else:



                    # instance_num = slide_embeddings.shape[0] // 5
                    # if instance_num == 0:
                    #     instance_num = 1
                    instance_num=num_crops

                    # slide_embeddings = slide_embeddings.view(instance_num, 5, 5, -1)
                    # print(f"instance_num is {instance_num}")
                    if instance_num < self.max_instances:
                        pad_size = self.max_instances - instance_num
                        pad = torch.zeros(
                            (pad_size, 5, slide_embeddings.shape[-1]),
                            dtype=slide_embeddings.dtype,
                            device=slide_embeddings.device,
                        )
                        final_embeddings = torch.cat([slide_embeddings, pad], dim=0)
                    


                    # final_embeddings = torch.mean(slide_embeddings, dim=1) #(max_instances, 5, embed_dim)
            else:

                if self.aggregation is None:
                    final_embeddings = torch.zeros(self.max_instances, self.embed_dim)
                else:
                    final_embeddings = torch.zeros(self.embed_dim)
                final_embeddings = final_embeddings.to(all_embeddings.device)
            
            final_embeddings_list.append(final_embeddings)
        

        if self.aggregation is None:

            return torch.stack(final_embeddings_list, dim=0)
        else:

            return torch.stack(final_embeddings_list, dim=0)

    def hierarchical_pooling(cell_embeddings, num_clusters=3, weighting="size"):
        """Hierarchical pooling by clustering cells into subpopulations for each channel"""

        _, num_channels, dim = cell_embeddings.shape
        well_embedding = np.zeros((num_channels, dim))

        for c in range(num_channels):
            channel_embeddings = cell_embeddings[:, c, :]

            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(channel_embeddings)

            subpop_embeddings = np.zeros((num_clusters, dim))
            subpop_sizes = np.zeros((num_clusters, 1))  # Ensure it has shape (num_clusters, 1)

            np.add.at(subpop_embeddings, cluster_labels, channel_embeddings)
            subpop_sizes[:, 0] = np.bincount(cluster_labels, minlength=num_clusters)

            mask = subpop_sizes[:, 0] > 0  # Extract the 1D boolean mask
            subpop_embeddings[mask] /= subpop_sizes[
                mask
            ]  # Perform element-wise division safely

            if weighting == "size":
                weights = subpop_sizes / np.sum(subpop_sizes)
            else:
                weights = np.ones_like(subpop_sizes) / num_clusters

            well_embedding[c, :] = np.sum(subpop_embeddings * weights, axis=0)

        return well_embedding


    def soft_kmeans_pooling(cell_embeddings, num_clusters=3):
        """Perform soft clustering using Gaussian Mixture Model (GMM) and weighted pooling."""
        _, num_channels, dim = cell_embeddings.shape
        well_embedding = np.zeros((num_channels, dim))

        for c in range(num_channels):
            channel_embeddings = cell_embeddings[:, c, :]

            # Soft clustering with GMM (alternative to K-Means)
            gmm = GaussianMixture(n_components=num_clusters, random_state=42)
            gmm.fit(channel_embeddings)
            soft_assignments = gmm.predict_proba(channel_embeddings)  # Soft cluster assignments

            # Compute cluster means
            subpop_embeddings = np.zeros((num_clusters, dim))
            for k in range(num_clusters):
                subpop_embeddings[k] = np.sum(
                    channel_embeddings * soft_assignments[:, k][:, None], axis=0
                )

            subpop_sizes = soft_assignments.sum(axis=0)[:, None]  # Soft counts for each cluster
            mask = subpop_sizes[:, 0] > 0  # Avoid division by zero
            subpop_embeddings[mask] /= subpop_sizes[mask]

            # Weighting based on entropy of assignments
            entropy_weights = -np.sum(
                soft_assignments * np.log(soft_assignments + 1e-8), axis=0
            )[:, None]
            entropy_weights /= np.sum(entropy_weights)

            well_embedding[c, :] = np.sum(subpop_embeddings * entropy_weights, axis=0)

        return well_embedding

    @property
    def dtype(self):
        return self.visual.transformer.resblocks[0].mlp.c_fc.weight.dtype

    def encode_mil(self, image):
        # print(f"image list shape is{image.shape}")
        # print(f"encode_mil received batch_images type: {type(image)}")
        # print(f"encode_mil batch_images length: {len(image)}")
        
        # if len(image) > 0:
            # print(f"First element type: {type(image[0])}")
            # if torch.is_tensor(image[0]):
                # print(f"First element shape: {image[0].shape}")

        # print(f"img emb shape is {image_embeddings.shape}")
        bag_feats = self.image_pool(image)  # (B, C, D)
        # bag_feats=image_embeddings
        # print(f"img pool shape is {bag_feats.shape}")
        return bag_feats

    def encode_image(self, image):

        image_feats = self.visual(image).type(self.dtype)  # (B, embed_dim)
        return image_feats

    def encode_text(self, text):
        out = self.text(
            input_ids=text["input_ids"], 
            attention_mask=text["attention_mask"]
        ).pooler_output.type(self.dtype)
        out = self.text_proj(out)
        return out

    def cross_modal_fusion(self, text_features, control_features):
        """
        ：TextFeatures
        Args:
            text_features: (batch_size, embed_dim)
            control_features: (batch_size, embed_dim)
        Returns:
            fused_text_features: (batch_size, embed_dim)
            fused_control_features: (batch_size, embed_dim)
        """

        text_residual = text_features
        control_residual = control_features
        

        for block in self.cross_modal_fusion_blocks:
            text_features, control_features = block(text_features, control_features)
        

        text_features = text_features + text_residual
        control_features = control_features + control_residual
        
        return text_features, control_features

    def forward(self, image, text, control=None,cross=True):

        image_features = self.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        

        text_features = self.compound(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)


        if control is None:
            if self.use_bias:
                return image_features, text_features, self.logit_scale, self.bias
            return image_features, text_features, self.logit_scale


        control_features = self.encode_image(control)   


        control_features = self.control_proj(control_features)
        control_features = control_features / control_features.norm(dim=-1, keepdim=True)
        

        fused_features = torch.cat([text_features, control_features], dim=-1)
        

        perturb_features = self.fusion_proj(fused_features)
        perturb_features = perturb_features / perturb_features.norm(dim=-1, keepdim=True)   

        if not cross:
            # print("not use cross")
            if self.use_bias:
                return image_features, perturb_features, self.logit_scale, self.bias

            return image_features, perturb_features, self.logit_scale


        image_features, perturb_features = self.cross_modal_fusion(
            image_features, perturb_features
        )
        

        if self.use_bias:
            return image_features, perturb_features, self.logit_scale, self.bias

        return image_features, perturb_features, self.logit_scale


class CLIP_ChannelViT(nn.Module):
    """Cloome with MLP and ChannelViT"""

    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        channels: int,
        depth: int,
        vision_heads: int,
        vision_patch_size: int,
        mlp_ratio: float,
        hcs: bool,
        # chem
        context_length: int,
        logit_scale: float = 14.3,
        learnable_logit_scale: bool = True,
    ):
        super().__init__()
        # swap ViT with ChannelViT
        if hcs:
            self.visual = ChannelViT_hcs(
                img_size=[image_resolution],
                patch_size=vision_patch_size,
                in_chans=channels,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=vision_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
        else:
            self.visual = ChannelViT(
                img_size=[image_resolution],
                patch_size=vision_patch_size,
                in_chans=channels,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=vision_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
        self.context_length = context_length

        # Load pre-trained text encoder from biomedical CLIP.
        self.text = BertModel.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        )
        self.text_proj = nn.Linear(768, 512)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale))
        self.logit_scale.requires_grad = learnable_logit_scale

        self.initialize_parameters()

    def initialize_parameters(self):
        if isinstance(self.visual, ChannelViT) or isinstance(self.visual, ChannelViT_hcs):
            for m in self.visual.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    @property
    def dtype(self):
        return self.visual.patch_embed.proj.weight.dtype

    def encode_image(self, image, extra_tokens):
        return self.visual(image, extra_tokens).type(self.dtype)

    def encode_text(self, text):
        out = self.text(
            input_ids=text["input_ids"], attention_mask=text["attention_mask"]
        ).pooler_output.type(self.dtype)
        out = self.text_proj(out)

        return out

    def forward(self, image, extra_tokens, text):
        image_features = self.encode_image(image, extra_tokens)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return image_features, text_features, self.logit_scale


class CLIP_ResNet(nn.Module):
    """Cloome from https://github.com/ml-jku/cloome/blob/main/src/clip/model.py"""

    def __init__(
        self,
        # Vision params
        vision_layers: List[int],
        embed_dim: int,
        input_channels: int,
        # text params
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        #
        logit_scale: float = 14.3,
        learnable_logit_scale: bool = True,
    ):
        super().__init__()

        self.context_length = context_length

        self.visual = ResNet(
            layers=vision_layers, output_dim=embed_dim, input_shape=input_channels
        )
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask,
        # with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except ValueError:
            return self.visual.fc.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding
        # (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return image_features, text_features, self.logit_scale


class Cloome(nn.Module):
    """Cloome from https://github.com/ml-jku/cloome/blob/main/src/clip/model.py"""

    def __init__(
        self,
        vision_layers: List[int],
        embed_dim: int,
        input_channels: int,
        input_size: int,
        molecule_layers: int,
        hidden_dim: int,
        logit_scale: float = 14.3,
        learnable_logit_scale: bool = True,
    ):
        super().__init__()

        self.visual = ResNet(
            layers=vision_layers, output_dim=embed_dim, input_shape=input_channels
        )
        self.chemical_encoder = MLP(
            input_dim=input_size,
            n_layers=molecule_layers,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
        )
        # Logit scales for the inner product in the InfoNCE loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale))
        self.logit_scale.requires_grad = learnable_logit_scale

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except ValueError:
            return self.visual.fc.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        return self.chemical_encoder(text.type(self.dtype))

    def forward(self, image, text):

        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features, self.logit_scale


class Cloome_old(nn.Module):
    """
    Cloome from https://github.com/ml-jku/cloome/blob/main/src/clip/model.py
    There are a lot of naming error in this version of cloome,
    keep it just to load ckpt from HF.
    """

    def __init__(
        self,
        vision_layers: List[int],
        embed_dim: int,
        input_channels: int,
        input_size: int,
        molecule_layers: int,
        hidden_dim: int,
        init_inv_tau: float = 14.3,
        learnable_inv_tau: bool = True,
    ):
        super().__init__()

        self.visual = ResNet(
            layers=vision_layers, output_dim=embed_dim, input_shape=input_channels
        )
        self.transformer = MLP(
            input_dim=input_size,
            n_layers=molecule_layers,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
        )
        # Logit scales for the inner product in the InfoNCE loss
        self.logit_inv_tau = nn.Parameter(torch.ones([]) * np.log(init_inv_tau))
        self.logit_inv_tau.requires_grad = learnable_inv_tau

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except ValueError:
            return self.visual.fc.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        return self.transformer(text.type(self.dtype))

    def forward(self, image, text):

        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features


class Cloome_phenom1(nn.Module):
    """Cloome variants from Molphenix"""

    def __init__(
        self,
        vision_width: int,
        vision_layers: int,
        vision_heads: int,
        embed_dim: int,
        input_size: int,
        molecule_layers: int,
        hidden_dim: int,
        logit_scale: float = 14.3,
        learnable_logit_scale: bool = True,
    ):
        super().__init__()

        self.visual = Transformer(vision_width, vision_layers, vision_heads)
        self.visual_proj = nn.Linear(vision_width, embed_dim)

        self.chemical_encoder = MLP(
            input_dim=input_size,
            n_layers=molecule_layers,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
        )
        # Logit scales for the inner product in the InfoNCE loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(logit_scale))
        self.logit_scale.requires_grad = learnable_logit_scale

    @property
    def dtype(self):
        return self.visual.resblocks[0].mlp.c_fc.weight.dtype

    def encode_image(self, image):
        image = self.visual(image).type(self.dtype)
        image = self.visual_proj(image)
        return image

    def encode_text(self, text):
        return self.chemical_encoder(text.type(self.dtype))

    def forward(self, image, text):

        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features, self.logit_scale


class Cloome_MPNN(nn.Module):
    """CLOOME variant: ResNet50 + MPNN++"""

    def __init__(
        self,
        vision_layers: List[int],
        input_channels: int,
        embed_dim: int,
        vision_width,
        vision_heads,
    ):
        super().__init__()

        self.visual = ResNet(
            layers=vision_layers, output_dim=embed_dim, input_shape=input_channels
        )

        # Load pre-trained chemical encoder
        # self.init_chem_encoder()
        self.mol_encoder = Transformer(vision_width, 1, vision_heads)

        self.chem_proj = nn.Linear(384, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except ValueError:
            return self.visual.fc.weight.dtype

    def init_chem_encoder(self):
        filename = "config_gps_10M_pcqm4m.yaml"

        cfg = load_yaml_config(
            os.path.join(constants.OUT_DIR, "configs/graphium_configs", filename)
        )
        cfg, accelerator_type = load_accelerator(cfg)
        datamodule = load_datamodule(cfg, accelerator_type)

        _, model_kwargs = load_architecture(
            cfg,
            in_dims=datamodule.in_dims,
        )
        self.chemical_encoder = FullGraphMultiTaskNetworkNew(**model_kwargs)

        # Load pretrained state dict
        new_state_dict = OrderedDict()
        state_dict = torch.load(
            os.path.join(
                constants.OUT_DIR, "results/mpnn/models/pcqm4mv2_mpnn_4layer.ckpt"
            ),
            map_location="cpu",
        )

        # Fix naming mismatches in keys
        for key, value in state_dict["state_dict"].items():
            new_key = key.replace("model.encoder_manager.", "encoder_manager.")
            new_key = new_key.replace("model.", "")
            new_key = new_key.replace("node_fully_connected", "node_model.fully_connected")
            new_key = new_key.replace("edge_fully_connected", "edge_model.fully_connected")
            new_state_dict[new_key] = value

        # Load the corrected state dict
        self.chemical_encoder.load_state_dict(new_state_dict, strict=True)

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        chem_features = convert_features_dtype(text)
        output = self.chemical_encoder(chem_features).type(self.dtype)

        return self.chem_proj(output)

    def encode_mols(self, mols):
        mols = self.mol_encoder(mols).type(self.dtype)
        mols = self.chem_proj(mols)
        return mols

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_mols(text)

        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        return image_features, text_features, self.logit_scale


def convert_features_dtype(feats):
    """Convert features to float32"""
    if isinstance(feats, torch.Tensor):
        feats = feats.to(torch.float32)
    elif isinstance(feats, (Data, Batch, dict)):
        for key, val in feats.items():
            if isinstance(val, torch.Tensor) and (val.is_floating_point()):
                feats[key] = val.to(dtype=torch.float32)
    return feats


def convert_ln_to_dyt(module):
    """Convert layernorm to dyt."""
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(layer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            layer.weight.data = layer.weight.data.half()
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.half()

        if isinstance(layer, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(layer, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(layer, name):
                attr = getattr(layer, name)
                if attr is not None:
                    if not isinstance(
                        attr, (nn.Sequential, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)
                    ):
                        attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, long_clip: bool):
    """Initialize CLIP with corresponding dict"""
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert (
            output_width ** 2 + 1
            == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks"))
    )
    input_channels = 3

    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        input_channels,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        long_clip,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    model.load_state_dict(state_dict)
    return model
