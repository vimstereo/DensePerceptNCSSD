import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from einops import rearrange, repeat

import math
import copy
try:
    from mamba_util import PatchMerging,SimplePatchMerging, Stem, SimpleStem, Mlp
except:
    from .mamba_util import PatchMerging, SimplePatchMerging, Stem, SimpleStem, Mlp
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

class tTensor(torch.Tensor):
    @property
    def shape(self):
        shape = super().shape
        return tuple([int(s) for s in shape])


to_ttensor = lambda *args: tuple([tTensor(x) for x in args]) if len(args) > 1 else tTensor(args[0])



class ConvFFN(nn.Module):

    def __init__(self, channels, expansion=2, drop=0.0):
        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Conv2d(self.dim1, self.dim2, 1, 1, 0)
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Conv2d(self.dim2, self.dim1, 1, 1, 0)
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x

class StandardAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.inner_dim = inner_dim


    def forward(self, x, H, W):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print(self.to_out(out).shape)
        return self.to_out(out)


class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_conv=3, #default to 3 for 2D
        conv_init=None,
        expand=2,
        headdim=64, #default to 64
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="silu", #default to silu
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=False, #default to False, for custom implementation
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        linear_attn_duality=False,
        d_state = 64,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        if ngroups == -1:
            ngroups = self.d_inner // self.headdim #equivalent to multi-head attention
        self.ngroups = ngroups
        print(self.d_inner,headdim)
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        #convert chunk_size to triton.language.int32
        self.chunk_size = chunk_size#torch.tensor(chunk_size,dtype=torch.int32)
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.ssd_positve_dA = kwargs.get('ssd_positve_dA', True) #default to False, ablation for linear attn duality
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias, **factory_kwargs) #
        self.proj_stereo = nn.Linear(self.d_inner, int(d_in_proj), bias=bias, **factory_kwargs) #

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state


        self.conv2d_l = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2d_r = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D_l = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D_l._no_weight_decay = True
        self.D_r = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D_r._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # modified from RMSNormGated to layer norm
        #assert RMSNormGated is not None
        #self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        #linear attention duality
        self.linear_attn_duality = linear_attn_duality
        self.kwargs = kwargs

    def non_casual_linear_attn(self, x, dt, A, B, C, D, H=None, W=None):
        '''
        non-casual attention duality of mamba v2
        x: (B, L, H, D), equivalent to V in attention
        dt: (B, L, nheads)
        A: (nheads) or (d_inner, d_state)
        B: (B, L, d_state), equivalent to K in attention
        C: (B, L, d_state), equivalent to Q in attention
        D: (nheads), equivalent to the skip connection
        '''

        batch, seqlen, head, dim = x.shape
        dstate = B.shape[2]
        V = x.permute(0, 2, 1, 3) # (B, H, L, D)
        dt = dt.permute(0, 2, 1) # (B, H, L)
        dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
        if self.ssd_positve_dA: dA = -dA

        V_scaled = V * dA
        K = B.view(batch, 1, seqlen, dstate)# (B, 1, L, D)
        if getattr(self, "__DEBUG__", False):
            A_mat = dA.cpu().detach().numpy()
            A_mat = A_mat.reshape(batch, -1, H, W)
            setattr(self, "__data__", dict(
                dA=A_mat, H=H, W=W, V=V,))

        if self.ngroups == 1:
            ## get kv via transpose K and V
            KV = K.transpose(-2, -1) @ V_scaled # (B, H, dstate, D)
            Q = C.view(batch, 1, seqlen, dstate)#.repeat(1, head, 1, 1)
            x = Q @ KV # (B, H, L, D)
            x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
        else:
            assert head % self.ngroups == 0
            dstate = dstate // self.ngroups
            K = K.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)
            V_scaled = V_scaled.view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
            Q = C.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)

            KV = K.transpose(-2, -1) @ V_scaled # (B, H//g, g, dstate, D)
            x = Q @ KV # (B, H//g, g, L, D)
            V_skip = (V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)).view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
            x = x + V_skip # (B, H//g, g, L, D)
            x = x.permute(0, 3, 1, 2, 4).flatten(2, 3).reshape(batch, seqlen, head, dim) # (B, L, H, D)
            x = x.contiguous()

        return x


    def forward(self, u, H, W, seq_idx=None):
        """
        u: (B,C,H,W)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape
        # print('u',u.shape)
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )

    
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        assert self.activation in ["silu", "swish"]
        

        #2D Convolution
        xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # xBC = self.act(self.conv2d(xBC))
        # xBC = xBC.permute(0, 2, 3, 1).view(batch, H*W, -1).contiguous()
        xBC1,xBC2 = torch.chunk(xBC,2,dim=0) 
        xBC1 = self.act(self.conv2d_l(xBC1))
        xBC2 = self.act(self.conv2d_r(xBC2))
        xBC1 = xBC1.permute(0, 2, 3, 1).view(batch//2, H*W, -1).contiguous()
        xBC2 = xBC2.permute(0, 2, 3, 1).view(batch//2, H*W, -1).contiguous()
        
        x1, B1, C1 = torch.split(xBC1, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x2, B2, C2 = torch.split(xBC2, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        dt1,dt2=torch.chunk(dt,2,dim=0)
        
        x1,x2, dt1,dt2, A, B1, B2, C1,C2 = to_ttensor(x1,x2,dt1,dt2, A, B1,B2, C1,C2)
    
        y1 = self.non_casual_linear_attn(
            rearrange(x1, "b l (h p) -> b l h p", p=self.headdim),
            dt1, A, B1, C1, self.D_l, H, W
        )
        y2 = self.non_casual_linear_attn(
            rearrange(x2, "b l (h p) -> b l h p", p=self.headdim),
            dt2, A, B2, C2, self.D_r, H, W
        )
            
        # print('after y',y.shape)       
        y1 = rearrange(y1, "b l h p -> b l (h p)")
        y2 = rearrange(y2, "b l h p -> b l (h p)")
        y1 = self.norm(y1)
        y2 = self.norm(y2)
        y=torch.concat((y1,y2),dim=0)
        y=y*z
        zxbcdt_stereo=self.proj_stereo(y)
        z_stereo, xBC_stereo, dt_stereo = torch.split(
            zxbcdt_stereo, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt_stereo = F.softplus(dt_stereo + self.dt_bias)
        xBC_stereo = xBC_stereo.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xBC_stereo = self.act(self.conv2d(xBC_stereo))
        xBC_stereo = xBC_stereo.permute(0, 2, 3, 1).view(batch, H*W, -1).contiguous()
        xs, Bs, Cs = torch.split(xBC_stereo, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        xs,dt_stereo, Bs, Cs = to_ttensor(xs,dt_stereo, Bs, Cs)

        ys = self.non_casual_linear_attn(
            rearrange(xs, "b l (h p) -> b l h p", p=self.headdim),
            dt_stereo, A, Bs, Cs, self.D, H, W
        )
        ys = rearrange(ys, "b l h p -> b l (h p)")
        ys = self.norm(ys)
        ys=ys*z_stereo

        ys = self.out_proj(ys)
        return ys





# class Mamba2(nn.Module):
#     def __init__(
#         self,
#         d_model,
#         d_conv=3, #default to 3 for 2D
#         conv_init=None,
#         expand=2,
#         headdim=64, #default to 64
#         ngroups=1,
#         A_init_range=(1, 16),
#         dt_min=0.001,
#         dt_max=0.1,
#         dt_init_floor=1e-4,
#         dt_limit=(0.0, float("inf")),
#         learnable_init_states=False,
#         activation="silu", #default to silu
#         bias=False,
#         conv_bias=True,
#         # Fused kernel and sharding options
#         chunk_size=256,
#         use_mem_eff_path=False, #default to False, for custom implementation
#         layer_idx=None,  # Absorb kwarg for general module
#         device=None,
#         dtype=None,
#         linear_attn_duality=False,
#         d_state = 64,
#         **kwargs
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.d_model = d_model
#         self.d_conv = d_conv
#         self.conv_init = conv_init
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.headdim = headdim
#         self.d_state = d_state
#         if ngroups == -1:
#             ngroups = self.d_inner // self.headdim #equivalent to multi-head attention
#         self.ngroups = ngroups
#         assert self.d_inner % self.headdim == 0
#         self.nheads = self.d_inner // self.headdim
#         self.dt_limit = dt_limit
#         self.learnable_init_states = learnable_init_states
#         self.activation = activation
#         #convert chunk_size to triton.language.int32
#         self.chunk_size = chunk_size#torch.tensor(chunk_size,dtype=torch.int32)
#         self.use_mem_eff_path = use_mem_eff_path
#         self.layer_idx = layer_idx
#         self.ssd_positve_dA = kwargs.get('ssd_positve_dA', True) #default to False, ablation for linear attn duality
#         # Order: [z, x, B, C, dt]
#         d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
#         self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias, **factory_kwargs) #

#         conv_dim = self.d_inner + 2 * self.ngroups * self.d_state


#         self.conv2d = nn.Conv2d(
#             in_channels=conv_dim,
#             out_channels=conv_dim,
#             groups=conv_dim,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             padding=(d_conv - 1) // 2,
#             **factory_kwargs,
#         )
#         if self.conv_init is not None:
#             nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
#         # self.conv1d.weight._no_weight_decay = True

#         if self.learnable_init_states:
#             self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
#             self.init_states._no_weight_decay = True

#         self.act = nn.SiLU()

#         # Initialize log dt bias
#         dt = torch.exp(
#             torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         )
#         dt = torch.clamp(dt, min=dt_init_floor)
#         # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         self.dt_bias = nn.Parameter(inv_dt)
#         # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
#         # name.endswith("bias") in param_grouping.py
#         self.dt_bias._no_weight_decay = True

#         # A parameter
#         assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
#         A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
#         A_log = torch.log(A).to(dtype=dtype)
#         self.A_log = nn.Parameter(A_log)
#         # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
#         self.A_log._no_weight_decay = True

#         # D "skip" parameter
#         self.D = nn.Parameter(torch.ones(self.nheads, device=device))
#         self.D._no_weight_decay = True

#         # modified from RMSNormGated to layer norm
#         #assert RMSNormGated is not None
#         #self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
#         self.norm = nn.LayerNorm(self.d_inner)
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

#         #linear attention duality
#         self.linear_attn_duality = linear_attn_duality
#         self.kwargs = kwargs

#     def non_casual_linear_attn(self, x, dt, A, B, C, D, H=None, W=None):
#         '''
#         non-casual attention duality of mamba v2
#         x: (B, L, H, D), equivalent to V in attention
#         dt: (B, L, nheads)
#         A: (nheads) or (d_inner, d_state)
#         B: (B, L, d_state), equivalent to K in attention
#         C: (B, L, d_state), equivalent to Q in attention
#         D: (nheads), equivalent to the skip connection
#         '''

#         batch, seqlen, head, dim = x.shape
#         dstate = B.shape[2]
#         V = x.permute(0, 2, 1, 3) # (B, H, L, D)
#         dt = dt.permute(0, 2, 1) # (B, H, L)
#         dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
#         if self.ssd_positve_dA: dA = -dA

#         V_scaled = V * dA
#         K = B.view(batch, 1, seqlen, dstate)# (B, 1, L, D)
#         if getattr(self, "__DEBUG__", False):
#             A_mat = dA.cpu().detach().numpy()
#             A_mat = A_mat.reshape(batch, -1, H, W)
#             setattr(self, "__data__", dict(
#                 dA=A_mat, H=H, W=W, V=V,))

#         if self.ngroups == 1:
#             ## get kv via transpose K and V
#             KV = K.transpose(-2, -1) @ V_scaled # (B, H, dstate, D)
#             Q = C.view(batch, 1, seqlen, dstate)#.repeat(1, head, 1, 1)
#             x = Q @ KV # (B, H, L, D)
#             x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
#             x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
#         else:
#             assert head % self.ngroups == 0
#             dstate = dstate // self.ngroups
#             K = K.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)
#             V_scaled = V_scaled.view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
#             Q = C.view(batch, 1, seqlen, self.ngroups, dstate).permute(0, 1, 3, 2, 4) # (B, 1, g, L, dstate)

#             KV = K.transpose(-2, -1) @ V_scaled # (B, H//g, g, dstate, D)
#             x = Q @ KV # (B, H//g, g, L, D)
#             V_skip = (V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)).view(batch, head//self.ngroups, self.ngroups, seqlen, dim) # (B, H//g, g, L, D)
#             x = x + V_skip # (B, H//g, g, L, D)
#             x = x.permute(0, 3, 1, 2, 4).flatten(2, 3).reshape(batch, seqlen, head, dim) # (B, L, H, D)
#             x = x.contiguous()

#         return x


#     def forward(self, u, H, W, seq_idx=None):
#         """
#         u: (B,C,H,W)
#         Returns: same shape as u
#         """
#         batch, seqlen, dim = u.shape

#         zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
#         A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
#         initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
#         dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)


#         z, xBC, dt = torch.split(
#             zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
#         )
#         dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
#         assert self.activation in ["silu", "swish"]


#         #2D Convolution
#         xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
#         xBC = self.act(self.conv2d(xBC))
#         xBC = xBC.permute(0, 2, 3, 1).view(batch, H*W, -1).contiguous()

#         # Split into 3 main branches: X, B, C
#         # These correspond to V, K, Q respectively in the SSM/attention duality
#         x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
#         x, dt, A, B, C = to_ttensor(x, dt, A, B, C)
#         x1,x2=torch.chunk(x,2,dim=0)
#         C1,C2=torch.chunk(C,2,dim=0)
#         B1,B2=torch.chunk(B,2,dim=0)
#         dt1,dt2=torch.chunk(dt,2,dim=0)
#         # print('x before',x1.shape)

#         # if self.linear_attn_duality:
#         y1 = self.non_casual_linear_attn(
#             rearrange(x1, "b l (h p) -> b l h p", p=self.headdim),
#             dt1, A, B1, C1, self.D, H, W
#         )
#         y2 = self.non_casual_linear_attn(
#             rearrange(x2, "b l (h p) -> b l h p", p=self.headdim),
#             dt2, A, B2, C2, self.D, H, W
#         )
            
#         # print('after y',y.shape)       
#         y1 = rearrange(y1, "b l h p -> b l (h p)")
#         y2 = rearrange(y2, "b l h p -> b l (h p)")
#         y1 = self.norm(y1)
#         y2 = self.norm(y2)
#         y=torch.concat((y1,y2),dim=0)
#         y=y*z
#         out = self.out_proj(y)
#         return out
        # z1,z2=torch.chunk(z,2,dim=0)
        # y1 = y1*z1
        # y2 = y2*z2
        
        # out1=torch.concat((y1,y2),dim=0)
        # out2=torch.concat((y2,y1),dim=0)
        # out1 = self.non_casual_linear_attn(
        #     rearrange(out1, "b l (h p) -> b l h p", p=self.headdim),
        #     dt, A, B, C, self.D, H, W
        # )
        # out2 = self.non_casual_linear_attn(
        #     rearrange(out2, "b l (h p) -> b l h p", p=self.headdim),
        #     dt, A, B, C, self.D, H, W
        # )
        # # print('out1',out1.shape)
        # b,c,x,y=out1.shape
        # out1=out1.view(b,c,x*y)
        # out2=out2.view(b,c,x*y)
        # out1 = self.out_proj(out1)
        # out2 = self.out_proj(out2)
        # out=(out1+out2)/2
        # # # Multiply "gate" branch and apply extra normalization layer
        # # y = self.norm(y, z)
        # print(out.shape)
        # return out


class VMAMBA2Block(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256,
                 linear_attn_duality=False, d_state = 64, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        if kwargs.get('attn_type', 'mamba2') == 'standard':
            self.attn = StandardAttention(dim=dim, heads=num_heads, dim_head=dim // num_heads, dropout=drop)
        elif kwargs.get('attn_type', 'mamba2') == 'mamba2':
            print('dim',dim,ssd_expansion,num_heads)
            self.attn = Mamba2(d_model=dim, expand=ssd_expansion, headdim= dim*ssd_expansion // num_heads,
                                ngroups=ssd_ngroups, chunk_size=ssd_chunk_size,
                                linear_attn_duality=linear_attn_duality, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H=None, W=None):
        # print(self.attn)
        B, L, C = x.shape
        # print('in vmam2block',x.shape)
        if H & W is None:
            H, W = self.input_resolution
            assert L == H * W, "input feature has wrong size"

        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)

        # SSD or Standard Attention
        x = self.attn(x, H, W)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    """ A basic MLLA layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256, linear_attn_duality=False, d_state=64, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            VMAMBA2Block(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer,
                      ssd_expansion=ssd_expansion, ssd_ngroups=ssd_ngroups, ssd_chunk_size=ssd_chunk_size,
                      linear_attn_duality=linear_attn_duality, d_state=d_state, **kwargs)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x, H=None, W=None):
        # print('basic layer',x.shape)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W)
            else:
                x = blk(x, H, W)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class VMAMBA2(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=64, depths=[2, 4, 12, 4], num_heads=[2, 4, 8, 16],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, use_checkpoint=False,
                 ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=512,
                 linear_attn_duality= True, d_state=128, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.simple_downsample = kwargs.get('simple_downsample', False)
        self.simple_patch_embed = kwargs.get('simple_patch_embed', False)
        self.attn_types = kwargs.get('attn_types', ['mamba2', 'mamba2', 'mamba2', 'standard'])
        if self.simple_patch_embed:
            self.patch_embed = SimpleStem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = Stem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        if self.simple_downsample:
            PatchMergingBlock = SimplePatchMerging
        else:
            PatchMergingBlock = PatchMerging
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution


        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs['attn_type'] = self.attn_types[i_layer]
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMergingBlock if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               ssd_expansion=ssd_expansion,
                               ssd_ngroups=ssd_ngroups,
                               ssd_chunk_size=ssd_chunk_size,
                               linear_attn_duality = linear_attn_duality,
                               d_state = d_state,
                               **kwargs)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

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
        return {'absolute_pos_embed'}

    @torch.no_grad()
    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        try:
            Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        except Exception as e:
            print('get exception', e)
            print('Error in flop_count, set to default value 1e9')
            return 1e9
        del model, input

        return sum(Gflops.values()) * 1e9

    def forward_features(self, x):
        H, W = x.shape[-2:]
        x = self.patch_embed(x)
        H, W = H//4, W//4 # downsampled by patch_embed

        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, H, W)
            H, W = H//2, W//2 # downsampled by layer

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        # print('forward feature',x.shape)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.head(x)
        return x


class Backbone_VMAMBA2(VMAMBA2):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, **kwargs):
        super().__init__(**kwargs)
        norm_layer = nn.LayerNorm
        self.dim_conv=nn.Conv2d(64, 256, kernel_size=1)
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.layers[i].dim)
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.head
        del self.norm
        del self.avgpool
        self.load_pretrained(pretrained,key=kwargs.get('key','model'))

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt} from {key}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")


    def forward(self, x):

        def layer_forward(l, x, H=None, W=None):
            for blk in l.blocks:
                x = blk(x, H, W)
            if l.downsample is not None:
                y = l.downsample(x, H, W)
            else:
                y = x
            return x, y
        print(type(x[1]))
        H, W = x.shape[-2:]
        # print('backbone before patch embed',x.shape)
        x = self.patch_embed(x)
        if self.simple_patch_embed:
            H, W = H//4, W//4
        else:
            H, W = int((H - 1) / 2) + 1, int((W - 1) / 2) + 1
            H, W = int((H - 1) / 2) + 1, int((W - 1) / 2) + 1
        outs = []
        # print('backbone',x.shape)
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x, H, W)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                B, L, C = out.shape
                out = out.view(B, H, W, C).permute(0, 3, 1, 2) # B, C, H, W
                outs.append(out.contiguous())
            #calculate H, W for next layer, with conv stride 3, stride 2 and padding 1
            H, W = int((H-1)/2)+1, int((W-1)/2)+1

        if len(self.out_indices) == 0:
            return x
        outs = self.dim_conv(outs[0])
        return outs