import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange
import numbers
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import math
from typing import Optional, Callable
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from itertools import repeat

def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x

def PhiTPhi_fun(x, PhiW):
    temp = F.conv2d(x, PhiW, padding=0,stride=32, bias=None)
    temp = F.conv_transpose2d(temp, PhiW, stride=32)
    return temp

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# Define Cross Attention Block
class blockNL(torch.nn.Module):
    def __init__(self, channels):
        super(blockNL, self).__init__()
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        
        self.norm_x = LayerNorm(1, 'WithBias')
        self.norm_z = LayerNorm(31, 'WithBias')

        self.t = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.p = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        self.v = nn.Conv2d(in_channels=self.channels+1, out_channels=self.channels+1, kernel_size=1, stride=1, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
        )

    def forward(self, x, z):
        
        x0 = self.norm_x(x)
        z0 = self.norm_z(z)
        
        z1 = self.t(z0)
        b, c, h, w = z1.shape
        z1 = z1.view(b, c, -1) # b, c, hw
        x1 = self.p(x0) # b, c, hw
        x1 = x1.view(b, c, -1)
        z1 = torch.nn.functional.normalize(z1, dim=-1)
        x1 = torch.nn.functional.normalize(x1, dim=-1)
        x_t = x1.permute(0, 2, 1) # b, hw, c
        att = torch.matmul(z1, x_t)
        att = self.softmax(att) # b, c, c
        
        z2 = self.g(z0)
        z_v = z2.view(b, c, -1)
        out_x = torch.matmul(att, z_v)
        out_x = out_x.view(b, c, h, w)
        out_x = self.w(out_x) + self.pos_emb(z2) + z
        y = self.v(torch.cat([x, out_x], 1))

        return y

# Define ISCA block
class Atten(torch.nn.Module):
    def __init__(self, channels):
        super(Atten, self).__init__()
               
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = LayerNorm(self.channels, 'WithBias')
        self.norm2 = LayerNorm(self.channels, 'WithBias')
        self.conv_q = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.conv_kv = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels*2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels*2, self.channels*2, kernel_size=3, stride=1, padding=1, groups=self.channels*2, bias=True)
        )
        self.conv_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        
    def forward(self, pre, cur):
        
        b, c, h, w = pre.shape
        pre_ln = self.norm1(pre)
        cur_ln = self.norm2(cur)
        q = self.conv_q(cur_ln)
        q = q.view(b, c, -1)
        k,v = self.conv_kv(pre_ln).chunk(2, dim=1)
        k = k.view(b, c, -1)
        v = v.view(b, c, -1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        att = torch.matmul(q, k.permute(0, 2, 1))
        att = self.softmax(att)
        out = torch.matmul(att, v).view(b, c, h, w)
        out = self.conv_out(out) + cur
        
        return out

# Define OCT
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.atten = Atten(31)
        self.nonlo = blockNL(channels=31)
        self.norm1 = LayerNorm(32, 'WithBias')
        self.norm2 = LayerNorm(32, 'WithBias')
        self.conv_forward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        self.conv_backward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        
    def forward(self, x, z_pre, z_cur, PhiWeight, PhiTb):
        
        z = self.atten(z_pre, z_cur)

        x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight)
        x_input = x + self.lambda_step * PhiTb
        x_input = self.nonlo(x_input, z)
    
        x = self.norm1(x_input)
        x_forward = self.conv_forward(x) + x_input
        x = self.norm2(x_forward)
        x_backward = self.conv_backward(x) + x_forward
        x_pred = x_input + x_backward

        return x_pred



class OCT_delete_ISCA_selfAttention_new_DDM(torch.nn.Module):
    def __init__(self, LayerNo, sensing_rate):
        super(OCT_delete_ISCA_selfAttention_new_DDM, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.patch_size = 32
        self.n_input = int(sensing_rate * 1024)

        for i in range(LayerNo):
            onelayer.append(BasicBlock_delete_ISCA_selfAttention_new_DDM())

        self.Phiweight = nn.Parameter(init.xavier_normal_(torch.Tensor(self.n_input, 1, self.patch_size, self.patch_size)))
        self.fcs = nn.ModuleList(onelayer)
        self.fe2 = nn.Conv2d(1, 31, 3, padding=1, bias=True)

    def forward(self, x):

        PhiTb = F.conv2d(x, self.Phiweight, stride=self.patch_size, padding=0, bias=None) # 64*1089*3*3
        PhiTb = F.conv_transpose2d(PhiTb, self.Phiweight, stride=self.patch_size)
        x = PhiTb
        z_cur = self.fe2(x)

        for i in range(self.LayerNo):
            x_dual = self.fcs[i](x, z_cur, self.Phiweight, PhiTb)
            x = x_dual[:, :1, :, :]
            z_cur = x_dual[:, 1:, :, :]

        x_final = x

        return x_final

class BasicBlock_delete_ISCA_selfAttention_new_DDM(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_delete_ISCA_selfAttention_new_DDM, self).__init__()

        self.traBlock = TraBlock_jun(32)
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.atten = Atten(31)
        self.nonlo = blockNL_delete_ISCA_selfAttention(channels=31)
        self.norm1 = LayerNorm(32, 'WithBias')
        self.norm2 = LayerNorm(32, 'WithBias')
        self.conv_forward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        self.conv_backward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )

    def forward(self, x, z_cur, PhiWeight, PhiTb):
        x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight)
        x_input = x + self.lambda_step * PhiTb
        x_input = self.nonlo(x_input, z_cur)

        # x = self.norm1(x_input)
        # x_forward = self.conv_forward(x) + x_input
        # x = self.norm2(x_forward)
        # x_backward = self.conv_backward(x) + x_forward
        # x_pred = x_input + x_backward
        x_pred = self.traBlock(x_input)

        return x_pred

class TraBlock_jun(nn.Module):
    def __init__(self, dim):
        super(TraBlock_jun, self).__init__()

        self.tra1 = TraBlock11(dim)
        self.tra2 = TraBlock22(dim)
        self.conv1 = nn.Conv2d(dim * 3, dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        x1 = self.tra1(x)
        x2 = self.tra2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)

        return x

class TraBlock11(nn.Module):
    def __init__(self, dim):
        super(TraBlock11, self).__init__()

        self.res1 = ResBlock(dim)

    def forward(self, x):
        x = self.res1(x)

        return x


class TraBlock22(nn.Module):
    def __init__(self, dim):
        super(TraBlock22, self).__init__()

        self.dowm = nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, stride=2, bias=True)
        self.res1 = ResBlock(dim * 2)

    def forward(self, x):
        x = self.res1(self.dowm(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')

        return x


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class blockNL_delete_ISCA_selfAttention(torch.nn.Module):
    def __init__(self, channels):
        super(blockNL_delete_ISCA_selfAttention, self).__init__()
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)

        self.norm_x = LayerNorm(1, 'WithBias')
        self.norm_z = LayerNorm(31, 'WithBias')

        self.t = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.p = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        self.v = nn.Conv2d(in_channels=self.channels + 1, out_channels=self.channels + 1, kernel_size=1, stride=1,
                           bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=self.channels),
        )

    def forward(self, x, z):
        # x0 = self.norm_x(x)
        z0 = self.norm_z(z)

        z1 = self.t(z0)
        b, c, h, w = z1.shape
        z1 = z1.view(b, c, -1)  # b, c, hw
        # x1 = self.p(x0)  # b, c, hw
        # x1 = x1.view(b, c, -1)
        z1 = torch.nn.functional.normalize(z1, dim=-1)
        # x1 = torch.nn.functional.normalize(x1, dim=-1)
        # x_t = x1.permute(0, 2, 1)  # b, hw, c
        x_t = z1.permute(0, 2, 1)  # b, hw, c
        att = torch.matmul(z1, x_t)
        att = self.softmax(att)  # b, c, c

        z2 = self.g(z0)
        z_v = z2.view(b, c, -1)
        out_x = torch.matmul(att, z_v)
        out_x = out_x.view(b, c, h, w)
        out_x = self.w(out_x) + self.pos_emb(z2) + z
        y = self.v(torch.cat([x, out_x], 1))

        return y


class BasicBlock_new_DDM(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_new_DDM, self).__init__()

        self.traBlock = TraBlock_jun(32)
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.atten = Atten(31)
        self.nonlo = blockNL_new_DDM(channels=31)
        self.norm1 = LayerNorm(32, 'WithBias')
        self.norm2 = LayerNorm(32, 'WithBias')
        self.conv_forward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        self.conv_backward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )

    def forward(self, x, z_cur, PhiWeight, PhiTb):

        x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight)
        x_input = x + self.lambda_step * PhiTb
        x_input = self.nonlo(x_input, z_cur)

        # x = self.norm1(x_input)
        # x_forward = self.conv_forward(x) + x_input
        # x = self.norm2(x_forward)
        # x_backward = self.conv_backward(x) + x_forward
        # x_pred = x_input + x_backward
        x_pred = self.traBlock(x_input)

        return x_pred

class blockNL_new_DDM(torch.nn.Module):
    def __init__(self, channels):
        super(blockNL_new_DDM, self).__init__()
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)

        self.norm_x = LayerNorm(1, 'WithBias')
        self.norm_z = LayerNorm(31, 'WithBias')

        self.t = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.p = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        self.v = nn.Conv2d(in_channels=self.channels + 1, out_channels=self.channels + 1, kernel_size=1, stride=1,
                           bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=self.channels),
        )

    def forward(self, x, z):
        # x0 = self.norm_x(x)
        z0 = self.norm_z(z)

        z1 = self.t(z0)
        b, c, h, w = z1.shape
        z1 = z1.view(b, c, -1)  # b, c, hw
        # x1 = self.p(x0)  # b, c, hw
        # x1 = x1.view(b, c, -1)
        z1 = torch.nn.functional.normalize(z1, dim=-1)
        # x1 = torch.nn.functional.normalize(x1, dim=-1)
        # x_t = x1.permute(0, 2, 1)  # b, hw, c
        x_t = z1.permute(0, 2, 1)  # b, hw, c
        att = torch.matmul(z1, x_t)
        att = self.softmax(att)  # b, c, c

        z2 = self.g(z0)
        z_v = z2.view(b, c, -1)
        out_x = torch.matmul(att, z_v)
        out_x = out_x.view(b, c, h, w)
        out_x = self.w(out_x) + self.pos_emb(z2) + z
        y = self.v(torch.cat([x, out_x], 1))

        return y


class BasicBlock_new_FFN(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_new_FFN, self).__init__()

        self.traBlock = TraBlock_jun(32)
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.atten = Atten(31)
        self.nonlo = blockNL_new_DDM(channels=31)
        self.norm1 = LayerNorm(32, 'WithBias')
        self.norm2 = LayerNorm(32, 'WithBias')
        self.conv_forward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        self.conv_backward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )

    def forward(self, x, z_cur, PhiWeight, PhiTb):

        x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight)
        x_input = x + self.lambda_step * PhiTb
        x_input = self.nonlo(x_input, z_cur)

        x = self.norm1(x_input)
        x_forward = self.conv_forward(x) + x_input
        x = self.norm2(x_forward)
        x_backward = self.conv_backward(x) + x_forward
        x_pred = x_input + x_backward
        # x_pred = self.traBlock(x_input)

        return x_pred


class BasicBlock_new_FFN2(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_new_FFN2, self).__init__()

        self.traBlock = TraBlock_jun(32)
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.atten = Atten(31)
        self.nonlo = blockNL_new_DDM(channels=31)
        self.norm1 = LayerNorm(32, 'WithBias')
        self.norm2 = LayerNorm(32, 'WithBias')
        self.conv_forward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        self.conv_backward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )

    def forward(self, x, z_cur, PhiWeight, PhiTb):

        x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight)
        x_input = x + self.lambda_step * PhiTb
        x_input = self.nonlo(x_input, z_cur)

        x = self.norm1(x_input)
        x_forward = self.conv_forward(x) + x_input
        x = self.norm2(x_forward)
        x_backward = self.conv_backward(x) + x_forward
        x_pred = x_input + x_backward
        # x_pred = self.traBlock(x_input)



class BasicBlock_ffn_new(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_ffn_new, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.nonlo = blockNL_ffn_new(channels=31)
        self.norm1 = LayerNorm(32, 'WithBias')
        # self.norm2 = LayerNorm(32, 'WithBias')
        self.conv_forward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        # self.conv_backward = nn.Sequential(
        #     nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
        #     nn.GELU(),
        #     nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
        #     nn.GELU(),
        #     nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        # )

    def forward(self, x, z_cur, PhiWeight, PhiTb):

        x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight)
        x_input = x + self.lambda_step * PhiTb
        x_input = self.nonlo(x_input, z_cur)

        x = self.norm1(x_input)
        x_forward = self.conv_forward(x) + x_input
        # x = self.norm2(x_forward)
        # x_backward = self.conv_backward(x) + x_forward
        # x_pred = x_input + x_backward
        x_pred = x_input + x_forward

        return x_pred

class blockNL_ffn_new_180(torch.nn.Module):
    def __init__(self, channels):
        super(blockNL_ffn_new_180, self).__init__()
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)

        # self.norm_x = LayerNorm(1, 'WithBias')
        self.norm_z = LayerNorm(47, 'WithBias')

        self.t = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        # self.p = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
        #     nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        # )
        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        self.v = nn.Conv2d(in_channels=self.channels + 1, out_channels=self.channels + 1, kernel_size=1, stride=1,
                           bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=self.channels),
        )

    def forward(self, x, z):
        # x0 = self.norm_x(x)
        z0 = self.norm_z(z)

        z1 = self.t(z0)
        b, c, h, w = z1.shape
        z1 = z1.view(b, c, -1)  # b, c, hw
        # x1 = self.p(x0)  # b, c, hw
        # x1 = x1.view(b, c, -1)
        z1 = torch.nn.functional.normalize(z1, dim=-1)
        # x1 = torch.nn.functional.normalize(x1, dim=-1)
        # x_t = x1.permute(0, 2, 1)  # b, hw, c
        x_t = z1.permute(0, 2, 1)
        att = torch.matmul(z1, x_t)
        att = self.softmax(att)  # b, c, c

        z2 = self.g(z0)
        z_v = z2.view(b, c, -1)
        out_x = torch.matmul(att, z_v)
        out_x = out_x.view(b, c, h, w)
        out_x = self.w(out_x) + self.pos_emb(z2) + z
        y = self.v(torch.cat([x, out_x], 1))

        return y


class blockNL_ffn_new(torch.nn.Module):
    def __init__(self, channels):
        super(blockNL_ffn_new, self).__init__()
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)

        # self.norm_x = LayerNorm(1, 'WithBias')
        self.norm_z = LayerNorm(31, 'WithBias')

        self.t = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        # self.p = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
        #     nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        # )
        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        self.v = nn.Conv2d(in_channels=self.channels + 1, out_channels=self.channels + 1, kernel_size=1, stride=1,
                           bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=self.channels),
        )

    def forward(self, x, z):
        # x0 = self.norm_x(x)
        z0 = self.norm_z(z)

        z1 = self.t(z0)
        b, c, h, w = z1.shape
        z1 = z1.view(b, c, -1)  # b, c, hw
        # x1 = self.p(x0)  # b, c, hw
        # x1 = x1.view(b, c, -1)
        z1 = torch.nn.functional.normalize(z1, dim=-1)
        # x1 = torch.nn.functional.normalize(x1, dim=-1)
        # x_t = x1.permute(0, 2, 1)  # b, hw, c
        x_t = z1.permute(0, 2, 1)
        att = torch.matmul(z1, x_t)
        att = self.softmax(att)  # b, c, c

        z2 = self.g(z0)
        z_v = z2.view(b, c, -1)
        out_x = torch.matmul(att, z_v)
        out_x = out_x.view(b, c, h, w)
        out_x = self.w(out_x) + self.pos_emb(z2) + z
        y = self.v(torch.cat([x, out_x], 1))

        return y



class BasicBlock_ffn_new2(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_ffn_new2, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.nonlo = blockNL_ffn_new(channels=31)
        self.norm1 = LayerNorm(32, 'WithBias')
        self.norm2 = LayerNorm(32, 'WithBias')
        self.conv_forward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        self.conv_backward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )

    def forward(self, x, z_cur, PhiWeight, PhiTb):

        x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight)
        x_input = x + self.lambda_step * PhiTb
        x_input = self.nonlo(x_input, z_cur)

        x = self.norm1(x_input)
        x_forward = self.conv_forward(x) + x_input
        x = self.norm2(x_forward)
        x_backward = self.conv_backward(x) + x_forward
        x_pred = x_input + x_backward

        return x_pred


class BasicBlock_ffn_new3(torch.nn.Module):
    def __init__(self):
        super(BasicBlock_ffn_new3, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.nonlo = blockNL_ffn_new(channels=31)
        self.norm1 = LayerNorm(32, 'WithBias')
        # self.norm2 = LayerNorm(32, 'WithBias')
        self.conv_forward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        # self.conv_backward = nn.Sequential(
        #     nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
        #     nn.GELU(),
        #     nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
        #     nn.GELU(),
        #     nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        # )

    def forward(self, x, z_cur, PhiWeight, PhiTb):

        x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight)
        x_input = x + self.lambda_step * PhiTb
        x_input = self.nonlo(x_input, z_cur)

        x = self.norm1(x_input)
        x_forward = self.conv_forward(x) + x_input
        # x = self.norm2(x_forward)
        # x_backward = self.conv_backward(x) + x_forward
        # x_pred = x_input + x_backward
        # x_pred = x_input + x_forward

        return x_forward

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError


        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D


    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out



class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=mlp_ratio,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale= nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))


    def forward(self, input, x_size):
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input*self.skip_scale + self.drop_path(self.self_attention(x))
        x = x*self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        patches_resolution = [img_size[0] // patch_size[0], img_size[0] // patch_size[0]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            # self.norm = norm_layer(embed_dim)
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops

class CAB(nn.Module):
    # compress_ratio=6 for light SR and compress_ratio=3 for classic SR
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=6,squeeze_factor=30):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x

    def flops(self):
        flops = 0
        return flops



class WindowAttention(nn.Module):
    r"""
    Shifted Window-based Multi-head Self-Attention

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        r"""
        Args:
            qkv: Input query, key, and value tokens with shape of (num_windows*b, n, c*3)
            rpi: Relative position index
            mask (0/-inf):  Mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'

class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt):
        B, L, C = x.shape
        K = 1 # mambairV2 needs noly 1 scan
        xs = x.permute(0,2,1).view(B,1,C,L).contiguous() # B, 1, C ,L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) + prompt # (b, k, d_state, l)  our ASE here!
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor,prompt, **kwargs):
        b,l,c = prompt.shape
        prompt = prompt.permute(0,2,1).contiguous().view(b,1,c,l)
        y = self.forward_core(x,prompt) # [B, L, C]
        y = y.permute(0,2,1).contiguous()
        return y


class ASSM(nn.Module):
    def __init__(self, dim, d_state, input_resolution, num_tokens=64, inner_rank=128, mlp_ratio = 2.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim*self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden,d_state=self.d_state,expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )


        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48]
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim//3),
            nn.GELU(),
            nn.Linear(self.dim//3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )



    def forward(self, x, x_size, token):
        B, n, C = x.shape
        H, W = x_size

        full_embedding = self.embeddingB.weight @ token.weight  # [128, C]

        pred_route = self.route(x)  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x = x.permute(0,2,1).reshape(B,C,H,W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B,cc,-1).contiguous().permute(0,2,1) # b,n,c

        semantic_x = semantic_neighbor(x, x_sort_indices)
        y = self.selectiveScan(semantic_x,prompt)
        y = self.out_proj(self.out_norm(y))
        x = semantic_neighbor(y, x_sort_indices_reverse)

        return x

def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


class AttentiveLayer(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 is_last=False,
                 ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.num_tokens = num_tokens
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.is_last = is_last
        self.inner_rank = inner_rank

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        layer_scale = 1e-4
        self.scale1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.win_mhsa = WindowAttention(
            self.dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.assm = ASSM(
            self.dim,
            d_state,
            input_resolution=input_resolution,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio
        )

        mlp_hidden_dim = int(dim * self.mlp_ratio)

        self.convffn1 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, )
        self.convffn2 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, )

        # uncomment here if you need to test on lightSR
        # self.convffn1 = GatedMLP(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim)
        # self.convffn2 =  GatedMLP(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim)

        self.embeddingA = nn.Embedding(self.inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)

    def forward(self, x, x_size, params):
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c

        # part1: Window-MHSA
        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)
        qkv = qkv.reshape(b, h, w, c3)
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = params['attn_mask']
        else:
            shifted_qkv = qkv
            attn_mask = None
        x_windows = window_partition(shifted_qkv, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c3)
        attn_windows = self.win_mhsa(x_windows, rpi=params['rpi_sa'], mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        x_win = attn_x.view(b, n, c) + shortcut
        x_win = self.convffn1(self.norm2(x_win), x_size) + x_win
        x = shortcut * self.scale1 + x_win

        # part2: Attentive State Space
        shortcut = x
        x_aca = self.assm(self.norm3(x), x_size, self.embeddingA) + x
        x = x_aca + self.convffn2(self.norm4(x_aca), x_size)
        x = shortcut * self.scale2 + x

        return x


class AttentiveLayer_reserve_yellow(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 is_last=False,
                 ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        layer_scale = 1e-4
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

        self.assm = ASSM(
            self.dim,
            d_state,
            input_resolution=input_resolution,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio
        )

        mlp_hidden_dim = int(dim * self.mlp_ratio)

        self.convffn2 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, )

        # uncomment here if you need to test on lightSR
        # self.convffn1 = GatedMLP(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim)
        # self.convffn2 =  GatedMLP(in_features=dim,hidden_features=mlp_hidden_dim,out_features=dim)

        self.embeddingA = nn.Embedding(self.inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)

    def forward(self, x, x_size, params):

        # part2: Attentive State Space
        shortcut = x
        x_aca = self.assm(self.norm3(x), x_size, self.embeddingA) + x
        x = x_aca + self.convffn2(self.norm4(x_aca), x_size)
        x = shortcut * self.scale2 + x

        return x


class BasicBlock_ffn_new4_mambaV3(torch.nn.Module):
    def __init__(self,
                 input_resolution,
                 d_state=8,
                 num_heads=4,
                 window_size=16,
                 inner_rank=32,
                 num_tokens=64,
                 convffn_kernel_size=5,
                 qkv_bias=True,
                 img_size=64,
                 patch_size=1,
                 embed_dim=32,
                 norm_layer=nn.LayerNorm,
                 drop_rate=0.,
                 patch_norm=True,
                 mlp_ratio=2.,
                 ):
        super(BasicBlock_ffn_new4_mambaV3, self).__init__()

        self.window_size = window_size
        self.input_resolution = input_resolution
        self.patch_norm = patch_norm
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.nonlo = blockNL_ffn_new(channels=31)
        self.num_features = embed_dim
        self.norm = norm_layer(self.num_features)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.blocks = AttentiveLayer_reserve_yellow(
                    dim=32,
                    d_state=d_state,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0,
                    inner_rank=inner_rank,
                    num_tokens=num_tokens,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    is_last=False,
                )

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.attBlocks = AttBlock(32)

    def calculate_rpi_sa(self):
        # calculate relative position index for SW-MSA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, z_cur, PhiWeight, PhiTb):

        x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight)
        x_input = x + self.lambda_step * PhiTb
        x_input = self.nonlo(x_input, z_cur)

        # 这一段是新的
        h_ori, w_ori = x_input.size()[-2], x_input.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad

        # self.mean = self.mean.type_as(x_forward2)
        # x_forward2 = (x_forward2 - self.mean) * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(x_input.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}
        x_size = (x_input.shape[2], x_input.shape[3])  # N,H,W,C
        x_forward2 = self.patch_embed(x_input)  # N,L,C
        # x_forward2 = self.pos_drop(x_forward2)
        x_forward2 = self.blocks(x_forward2, x_size, params)
        x_forward2 = self.norm(x_forward2)
        x_forward2 = self.patch_unembed(x_forward2, x_size)
        # x_forward2 = x_forward2 / self.img_range + self.mean
        # 这一段是新的

        x_backward = self.attBlocks(x_forward2) +  x_input  #SSA
        return x_backward

class MTCSNet(torch.nn.Module):
    def __init__(self, LayerNo, sensing_rate, patch_norm=True):
        super(MTCSNet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.patch_size = 32
        self.n_input = int(sensing_rate * 1024)
        img_size = 64,
        patch_size = 1,
        in_chans = 3,
        embed_dim=32,
        patch_norm = True,
        norm_layer = nn.LayerNorm,
        self.patch_norm = patch_norm

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        for i in range(LayerNo):
            onelayer.append(BasicBlock_ffn_new4_mambaV3(input_resolution=(patches_resolution[0], patches_resolution[1])))

        self.Phiweight = nn.Parameter(init.xavier_normal_(torch.Tensor(self.n_input, 1, self.patch_size, self.patch_size)))
        self.fcs = nn.ModuleList(onelayer)
        self.fe2 = nn.Conv2d(1, 31, 3, padding=1, bias=True)

    def forward(self, x):

        PhiTb = F.conv2d(x, self.Phiweight, stride=self.patch_size, padding=0, bias=None) # 64*1089*3*3
        PhiTb = F.conv_transpose2d(PhiTb, self.Phiweight, stride=self.patch_size)
        x = PhiTb
        z_cur = self.fe2(x)

        for i in range(self.LayerNo):
            x_dual = self.fcs[i](x, z_cur, self.Phiweight, PhiTb)
            x = x_dual[:, :1, :, :]
            z_cur = x_dual[:, 1:, :, :]
        x_final = x

        return x_final


class AttBlock(nn.Module):
    def __init__(self, dim):
        super(AttBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, bias=True, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.scale = dim ** -0.5
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv1(x)
        x = x.reshape(B, C, H // 32, 32, W // 32, 32).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, 32, 32)
        x1 = self.conv2(x).reshape(-1, C, 32 * 32)
        x2 = self.conv3(x).reshape(-1, C, 32 * 32).transpose(1, 2)
        att = (x2 @ x1) * self.scale
        att = att.softmax(dim=1)
        x = (x.reshape(-1, C, 32 * 32) @ att).reshape(-1, C, 32, 32)
        x = self.conv4(x)
        x = x.reshape(B, H // 32, W // 32, C, 32, 32).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        return x
