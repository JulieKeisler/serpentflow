"""
Model definitions for SerpentFlow.

Includes:
    - Residual blocks, attention blocks, up/downsampling
    - Timestep embeddings for continuous flows
    - UNetFlow generative model
    - BinaryImageClassifier for cutoff frequency determination
"""

from dataclasses import dataclass
from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def normalization(ch):
    # GroupNorm simple et robuste
    for g in (32,16,8,4,2,1):
        if ch % g == 0: return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)

def timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2: emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

class TimestepBlock(nn.Module):
    """Module whose forward takes timestep embeddings as second argument."""
    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential module that passes timestep embeddings to children that support it."""
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv=True, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv=True, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            self.op = nn.Conv2d(channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        return self.op(x)


class ResBlock(TimestepBlock):
    """Residual block with optional timestep embedding"""
    def __init__(self, channels, emb_channels, dropout=0.0, out_channels=None, use_scale_shift_norm=False, up=False, down=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1)
        )

        if up:
            self.h_upd = Upsample(channels, use_conv=False)
            self.x_upd = Upsample(channels, use_conv=False)
        elif down:
            self.h_upd = Downsample(channels, use_conv=False)
            self.x_upd = Downsample(channels, use_conv=False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels)
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        if isinstance(self.in_layers, nn.Sequential):
            h = self.in_layers(x)
        h_emb = self.emb_layers(emb).type(h.dtype)
        while len(h_emb.shape) < len(h.shape):
            h_emb = h_emb[..., None, None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(h_emb, 2, dim=1)
            h = self.out_layers[0](h) * (1 + scale) + shift
            h = self.out_layers[1:](h)
        else:
            h = h + h_emb
            h = self.out_layers(h)
        return self.skip_connection(self.x_upd(x)) + h


class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention(num_heads)
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x_in = x
        x = x.reshape(b, c, -1)
        h = self.norm(x)
        h = self.qkv(h)
        h = self.attention(h)
        h = self.proj_out(h)
        return (x_in + h.reshape(b, c, *spatial))


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).reshape(bs * self.n_heads, ch, length),
            (k * scale).reshape(bs * self.n_heads, ch, length)
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        h = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return h.reshape(bs, -1, length)
class UNetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t):
        out = self.model(x, t)
        return getattr(out, "sample", out)

    def __getattr__(self, name):
        if name == "model":
            return super().__getattr__(name)
        return getattr(self.model, name)

@dataclass(eq=False)
class UNetFlow(nn.Module):
    C_in: int
    C_out: int
    base_ch: int = 128
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int] = (1, 2)
    dropout: float = 0.0
    ch_mult: Tuple[int] = (1, 2, 4)
    conv_resample: bool = True
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False

    def __post_init__(self):
        super().__init__()
    
        self.time_embed_dim = self.base_ch * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.base_ch, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        ch = input_ch = self.ch_mult[0] * self.base_ch
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(self.C_in, ch, 3, padding=1))])
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.ch_mult):
            for _ in range(self.num_res_blocks):
                layers = [ResBlock(ch, self.time_embed_dim, self.dropout, out_channels=mult * self.base_ch, use_scale_shift_norm=self.use_scale_shift_norm)]
                ch = mult * self.base_ch
                if ds in self.attention_resolutions:
                    layers.append(AttentionBlock(ch))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(self.ch_mult) - 1:
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, self.conv_resample, out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, self.time_embed_dim, self.dropout, use_scale_shift_norm=self.use_scale_shift_norm),
            AttentionBlock(ch),
            ResBlock(ch, self.time_embed_dim, self.dropout, use_scale_shift_norm=self.use_scale_shift_norm)
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(self.ch_mult))):
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich, self.time_embed_dim, self.dropout, out_channels=mult * self.base_ch, use_scale_shift_norm=self.use_scale_shift_norm)]
                ch = mult * self.base_ch
                if ds in self.attention_resolutions:
                    layers.append(AttentionBlock(ch))
                if level and i == self.num_res_blocks:
                    layers.append(Upsample(ch, self.conv_resample))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, self.C_out, 3, padding=1))
        )

    def forward(self, x, timesteps):
        emb = self.time_embed(timestep_embedding(timesteps, self.base_ch).to(x))
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            if h.shape[-2:] != hs[-1].shape[-2:]:
                h = F.interpolate(h, size=hs[-1].shape[-2:], mode='nearest')
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)


# -------------------------------
# Binary Classifier
# -------------------------------

class BinaryImageClassifier(nn.Module):
    """Simple CNN binary classifier for low/high-frequency cutoff selection."""
    def __init__(self, in_channels, sigmoid=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 1)
        self.sigmoid = sigmoid

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        if self.sigmoid:
            return torch.sigmoid(self.fc(x))
        else:
            return self.fc(x)
