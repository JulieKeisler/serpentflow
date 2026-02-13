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


def make_attention(ch, ds, attn_res, attn_type):
    if ds not in attn_res or attn_type == "none":
        return nn.Identity()
    if attn_type == "self":
        return AttentionBlock(ch)
    if attn_type == "cbam":
        return CBAM(ch)
    if attn_type == "lite_gn_cbam":
        return LiteGN_CBAM(ch)
    # Dans make_attention, ajouter:
    if attn_type == "cross_channel":
        return CrossChannelAttention(ch)
    if attn_type == "local_cross":
        return LocalCrossAttention(ch, patch_size=5, num_heads=2)
    raise ValueError(attn_type)

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

# ============================================================
# Self-attention (QKV)
# ============================================================

class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = normalization(ch)
        self.qkv = nn.Conv1d(ch, ch * 3, 1)
        self.proj = zero_module(nn.Conv1d(ch, ch, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        x = x.view(b, c, -1)
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)
        h = (attn @ v.transpose(1, 2)).transpose(1, 2)
        return x_in + self.proj(h).view(b, c, h.shape[-1] // w, w)

# ============================================================
# CBAM variants
# ============================================================

class CBAM(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x)
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return x * self.sa(torch.cat([avg, mx], 1))

class LiteGN_CBAM(nn.Module):
    def __init__(self, ch, groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(min(groups, ch), ch)
        self.conv1d = nn.Conv1d(1, 1, 3, padding=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.gn(x).mean((2, 3))          # (B,C)
        y = self.conv1d(y.unsqueeze(1)).sigmoid().squeeze(1)
        return x * y[:, :, None, None]

class CrossChannelAttention(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = normalization(ch)
        self.qkv = nn.Linear(ch, ch * 3)
        self.proj = nn.Linear(ch, ch)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, H*W, C]
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        x_norm = self.norm(x_flat.permute(0,2,1)).permute(0,2,1)  # normalisation compatible
        qkv = self.qkv(x_norm).chunk(3, dim=-1)
        q, k, v = qkv
        # multi-head
        q = q.view(b, h*w, self.num_heads, c // self.num_heads).transpose(1,2)
        k = k.view(b, h*w, self.num_heads, c // self.num_heads).transpose(1,2)
        v = v.view(b, h*w, self.num_heads, c // self.num_heads).transpose(1,2)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(c // self.num_heads), dim=-1)
        out = (attn @ v).transpose(1,2).reshape(b, h*w, c)
        out = self.proj(out)
        out = out.reshape(b, h, w, c).permute(0,3,1,2)  # back to [B,C,H,W]
        return x + out

class LocalCrossAttention(nn.Module):
    def __init__(self, channels, patch_size=5, num_heads=2):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        ph, pw = self.patch_size, self.patch_size

        pad_h = (ph - h % ph) % ph
        pad_w = (pw - w % pw) % pw
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        H, W = x_padded.shape[2], x_padded.shape[3]

        x_patches = x_padded.unfold(2, ph, ph).unfold(3, pw, pw)
        nH, nW = x_patches.shape[2], x_patches.shape[3]
        x_patches = x_patches.permute(0,2,3,4,5,1).contiguous().view(-1, ph*pw, c)

        x_norm = self.norm(x_patches)
        qkv = self.qkv(x_norm).chunk(3, dim=-1)
        q, k, v = qkv
        q = q.view(q.shape[0], q.shape[1], self.num_heads, c//self.num_heads).transpose(1,2)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, c//self.num_heads).transpose(1,2)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, c//self.num_heads).transpose(1,2)
        attn = torch.softmax(q @ k.transpose(-2,-1) / math.sqrt(c//self.num_heads), dim=-1)
        out = (attn @ v).transpose(1,2).reshape(x_patches.shape[0], ph*pw, c)
        out = self.proj(out)

        out = out.view(b, nH, nW, ph, pw, c).permute(0,5,1,3,2,4).contiguous()
        out = out.view(b, c, H, W)

        H_out, W_out = out.shape[2], out.shape[3]
        H_min, W_min = min(h, H_out), min(w, W_out)
        return x[:, :, :H_min, :W_min] + out[:, :, :H_min, :W_min]



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

# -------------------------------
# UNetFlow
# -------------------------------
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
    attention_type: str = "none"
    time_factor: int = 4
    stats_dim: int = 2  # nombre de channels projetés des stats 1D

    def __post_init__(self):
        super().__init__()

        # Time embedding
        self.time_embed_dim = self.base_ch * self.time_factor
        self.time_embed = nn.Sequential(
            nn.Linear(self.base_ch, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

        # Input blocks
        ch = input_ch = self.ch_mult[0] * self.base_ch
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(self.C_in, ch, 3, padding=1))])
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.ch_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=mult * self.base_ch,
                        use_scale_shift_norm=self.use_scale_shift_norm
                    ),
                    make_attention(mult * self.base_ch, ds, self.attention_resolutions, self.attention_type)
                ]
                ch = mult * self.base_ch
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(self.ch_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, self.conv_resample, ch)
                    )
                )
                input_block_chans.append(ch)
                ds *= 2

        # Middle block avec stats_dim
        ch += self.stats_dim
        middle_layers = [
            ResBlock(ch, self.time_embed_dim, self.dropout, use_scale_shift_norm=self.use_scale_shift_norm)
        ]
        if ds in self.attention_resolutions:
            middle_layers.append(AttentionBlock(ch))
        middle_layers.append(
            ResBlock(ch, self.time_embed_dim, self.dropout, out_channels=ch, use_scale_shift_norm=self.use_scale_shift_norm)
        )
        self.middle_block = TimestepEmbedSequential(*middle_layers)

        # Output blocks
        self.output_blocks = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(self.ch_mult))):
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_embed_dim,
                        self.dropout,
                        out_channels=mult * self.base_ch,
                        use_scale_shift_norm=self.use_scale_shift_norm
                    ),
                    make_attention(mult * self.base_ch, ds, self.attention_resolutions, self.attention_type)
                ]
                ch = mult * self.base_ch

                if level and i == self.num_res_blocks:
                    layers.append(Upsample(ch, self.conv_resample))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Output layer
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, self.C_out, 3, padding=1))
        )

        # Linear pour stats
        if self.stats_dim > 0:
            self.stats_proj = nn.Linear(2 * self.C_in, self.stats_dim)

    def forward(self, x, timesteps, stats=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.base_ch).to(x))
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        # Injection des stats 1D avant middle block
        if stats is not None and self.stats_dim > 0:
            B, C, _ = stats.shape
            stats_flat = stats.view(B, C*2)
            stats_emb = self.stats_proj(stats_flat)  # [B, stats_dim]
            stats_emb = stats_emb[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])
            h = torch.cat([h, stats_emb], dim=1)
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