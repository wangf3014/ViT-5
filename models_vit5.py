import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

try:
    from flash_attn import flash_attn_qkvpacked_func
except ImportError:
    pass

from rope import VisionRotaryEmbedding

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., flash=True,
                 rope_size=0, rope_reg_size=0, num_registers=0, reg_theta=10000, qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.flash = flash
        self.num_registers = num_registers
        self.rope = VisionRotaryEmbedding(head_dim//2, rope_size) if rope_size > 0 else None
        self.rope_reg = VisionRotaryEmbedding(head_dim//2, rope_reg_size, theta=reg_theta) if rope_reg_size > 0 else None

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(self, x):
        B, N, C = x.shape
        reg_idx = N - self.num_registers

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)

        if self.qk_norm:
            qk_dtype = q.dtype
            q = self.q_norm(q).to(qk_dtype)
            k = self.k_norm(k).to(qk_dtype)

        if self.rope is not None:
            q = torch.cat((q[:, :1], self.rope(q[:, 1: reg_idx]), q[:, reg_idx:]), dim=1)
            k = torch.cat((k[:, :1], self.rope(k[:, 1: reg_idx]), k[:, reg_idx:]), dim=1)
        if self.rope_reg is not None:
            q = torch.cat((q[:, :1], q[:, 1: reg_idx], self.rope_reg(q[:, reg_idx:])), dim=1)
            k = torch.cat((k[:, :1], k[:, 1: reg_idx], self.rope_reg(k[:, reg_idx:])), dim=1)
        
        if self.flash:
            qkv = torch.stack([q, k, v], dim=2)
            x = flash_attn_qkvpacked_func(qkv).reshape(B, N, C)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.,
                 norm_layer=nn.LayerNorm, subln=False,):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp, init_values=1e-4,
                 flash=True, rope_size=0, rope_reg_size=0, reg_theta=10000, num_registers=0, qk_norm=False, layer_scale=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, flash=flash,
            rope_size=rope_size, rope_reg_size=rope_reg_size, num_registers=num_registers, qk_norm=qk_norm, reg_theta=reg_theta)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        if self.layer_scale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x        
    
class vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, ape=True,
                 block_layers=Block, Patch_layer=PatchEmbed, act_layer=nn.GELU, Attention_block=Attention, Mlp_block=Mlp,
                 init_scale=1e-4, flash=True, rope=False, num_registers=0, qk_norm=False, reg_theta=10000, layer_scale=True, **kwargs):
        super().__init__()       
        self.dropout_rate = drop_rate  
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_registers = num_registers

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, num_registers, embed_dim)) if num_registers > 0 else None

        rope_reg_size = int(num_registers ** 0.5)
        assert rope_reg_size ** 2 == num_registers, "num_registers must be a square number"

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) if ape else None

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale,
                flash=flash, rope_size=img_size // patch_size if rope else 0,
                rope_reg_size=rope_reg_size, num_registers=num_registers,
                qk_norm=qk_norm, reg_theta=reg_theta, layer_scale=layer_scale)
            for i in range(depth)])
           
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        
        trunc_normal_(self.cls_token, std=.02)
        if ape:
            trunc_normal_(self.pos_embed, std=.02)
        if num_registers > 0:
            trunc_normal_(self.reg_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'reg_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        registers = self.reg_token.expand(B, -1, -1) if self.reg_token is not None else None
        
        if self.pos_embed is not None:
            x = x + self.pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)
        if registers is not None:
            x = torch.cat((x, registers), dim=1)
            
        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):

        x = self.forward_features(x)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        return x

@register_model
def deit_small_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, flash=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block, **kwargs)
    return model

@register_model
def deit_base_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, flash=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block, **kwargs)
    return model

@register_model
def deit_large_patch16_LS(img_size=224, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, flash=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block, **kwargs)
    return model

@register_model
def vit5_small(img_size=224, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=False,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Block, rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model

@register_model
def vit5_base(img_size=224, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=False,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Block, rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model

@register_model
def vit5_large(img_size=224, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=False,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Block, rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model

@register_model
def vit5_xlarge(img_size=224, **kwargs):
    model = vit_models(
        img_size=img_size, patch_size=16, embed_dim=1152, depth=28, num_heads=16, mlp_ratio=4, qkv_bias=False, num_registers=4, flash=False,
        norm_layer=partial(RMSNorm, eps=1e-6), block_layers=Block, rope=True, rope_reg=True, reg_theta=100, qk_norm=True, **kwargs)
    return model