import torch
import torch.nn as nn

# 定義可變形卷積層
class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
            
            offset = self.p_conv(x)
            if self.modulation:
                m = torch.sigmoid(self.m_conv(x))

            dtype = offset.data.type()
            ks = self.kernel_size
            N = offset.size(1) // 2

            if self.padding:
                x = self.zero_padding(x)

            p = self._get_p(offset, dtype)
            p = p.contiguous().permute(0, 2, 3, 1)
            q_lt = p.detach().floor()
            q_rb = q_lt + 1

            q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
            q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
            q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
            q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

            p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

            g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
            g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
            g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
            g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

            x_q_lt = self._get_x_q(x, q_lt, N)
            x_q_rb = self._get_x_q(x, q_rb, N)
            x_q_lb = self._get_x_q(x, q_lb, N)
            x_q_rt = self._get_x_q(x, q_rt, N)

            x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                    g_rb.unsqueeze(dim=1) * x_q_rb + \
                    g_lb.unsqueeze(dim=1) * x_q_lb + \
                    g_rt.unsqueeze(dim=1) * x_q_rt

            if self.modulation:
                m = m.contiguous().permute(0, 2, 3, 1)
                m = m.unsqueeze(dim=1)
                m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
                x_offset *= m

            x_offset = self._reshape_x_offset(x_offset, ks)
            out = self.conv(x_offset)

            return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))

        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)
        p_n = self._get_p_n(N, dtype)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)

        index = q[..., :N]*padded_w + q[..., N:]
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)
        return x_offset

# 定義蛇形卷積層
class DSConv_pro(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=9, extend_scope=1.0, morph=0, if_offset=True, device="cuda"):
        super().__init__()

        if morph not in (0, 1):
            raise ValueError("morph should be 0 or 1.")

        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.morph = morph
        self.if_offset = if_offset
        self.device = torch.device(device)
        self.to(device)

        self.gn_offset = nn.GroupNorm(kernel_size, 2 * kernel_size)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size, 3, padding=1)

        self.dsc_conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(kernel_size, 1), padding=0)
        self.dsc_conv_y = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, kernel_size), padding=0)

    def forward(self, input):
        offset = self.offset_conv(input)
        offset = self.gn_offset(offset)
        offset = self.tanh(offset)

        y_coordinate_map, x_coordinate_map = get_coordinate_map_2D(offset=offset, morph=self.morph,
                                                                 extend_scope=self.extend_scope, device=self.device)
        deformed_feature = get_interpolated_feature(input, y_coordinate_map, x_coordinate_map)

        if self.morph == 0:
            output = self.dsc_conv_x(deformed_feature)
        elif self.morph == 1:
            output = self.dsc_conv_y(deformed_feature)

        output = self.gn(output)
        output = self.relu(output)

        return output



def get_coordinate_map_2D(offset, morph, extend_scope=1.0, device="cuda"):
    if morph not in (0, 1):
        raise ValueError("morph should be 0 or 1.")

    batch_size, _, width, height = offset.shape
    kernel_size = offset.shape[1] // 2
    center = kernel_size // 2
    device = torch.device(device)

    y_offset_, x_offset_ = torch.split(offset, kernel_size, dim=1)

    y_center_ = torch.arange(0, width, dtype=torch.float32, device=device)
    y_center_ = einops.repeat(y_center_, "w -> k w h", k=kernel_size, h=height)

    x_center_ = torch.arange(0, height, dtype=torch.float32, device=device)
    x_center_ = einops.repeat(x_center_, "h -> k w h", k=kernel_size, w=width)

    if morph == 0:
        y_spread_ = torch.zeros([kernel_size], device=device)
        x_spread_ = torch.linspace(-center, center, kernel_size, device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        y_offset_ = einops.rearrange(y_offset_, "b k w h -> k b w h")
        y_offset_new_ = y_offset_.detach().clone()

        y_offset_new_[center] = 0

        for index in range(1, center + 1):
            y_offset_new_[center + index] = (y_offset_new_[center + index - 1] + y_offset_[center + index])
            y_offset_new_[center - index] = (y_offset_new_[center - index + 1] + y_offset_[center - index])

        y_offset_new_ = einops.rearrange(y_offset_new_, "k b w h -> b k w h")

        y_new_ = y_new_.add(y_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b (w k) h")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b (w k) h")

    elif morph == 1:
        y_spread_ = torch.linspace(-center, center, kernel_size, device=device)
        x_spread_ = torch.zeros([kernel_size], device=device)

        y_grid_ = einops.repeat(y_spread_, "k -> k w h", w=width, h=height)
        x_grid_ = einops.repeat(x_spread_, "k -> k w h", w=width, h=height)

        y_new_ = y_center_ + y_grid_
        x_new_ = x_center_ + x_grid_

        y_new_ = einops.repeat(y_new_, "k w h -> b k w h", b=batch_size)
        x_new_ = einops.repeat(x_new_, "k w h -> b k w h", b=batch_size)

        x_offset_ = einops.rearrange(x_offset_, "b k w h -> k b w h")
        x_offset_new_ = x_offset_.detach().clone()

        x_offset_new_[center] = 0

        for index in range(1, center + 1):
            x_offset_new_[center + index] = (x_offset_new_[center + index - 1] + x_offset_[center + index])
            x_offset_new_[center - index] = (x_offset_new_[center - index + 1] + x_offset_[center - index])

        x_offset_new_ = einops.rearrange(x_offset_new_, "k b w h -> b k w h")

        x_new_ = x_new_.add(x_offset_new_.mul(extend_scope))

        y_coordinate_map = einops.rearrange(y_new_, "b k w h -> b w (h k)")
        x_coordinate_map = einops.rearrange(x_new_, "b k w h -> b w (h k)")

    return y_coordinate_map, x_coordinate_map



def get_interpolated_feature(input_feature, y_coordinate_map, x_coordinate_map, interpolate_mode="bilinear"):
   if interpolate_mode not in ("bilinear", "bicubic"):
       raise ValueError("interpolate_mode should be 'bilinear' or 'bicubic'.")

   y_max = input_feature.shape[-2] - 1
   x_max = input_feature.shape[-1] - 1

   y_coordinate_map_ = _coordinate_map_scaling(y_coordinate_map, origin=[0, y_max])
   x_coordinate_map_ = _coordinate_map_scaling(x_coordinate_map, origin=[0, x_max])

   y_coordinate_map_ = torch.unsqueeze(y_coordinate_map_, dim=-1)
   x_coordinate_map_ = torch.unsqueeze(x_coordinate_map_, dim=-1)

   grid = torch.cat([x_coordinate_map_, y_coordinate_map_], dim=-1)

   interpolated_feature = nn.functional.grid_sample(
       input=input_feature,
       grid=grid,
       mode=interpolate_mode,
       padding_mode="zeros",
       align_corners=True,
   )

   return interpolated_feature

def _coordinate_map_scaling(coordinate_map, origin, target=[-1, 1]):
   min, max = origin
   a, b = target

   coordinate_map_scaled = torch.clamp(coordinate_map, min, max)

   scale_factor = (b - a) / (max - min)
   coordinate_map_scaled = a + scale_factor * (coordinate_map_scaled - min)

   return coordinate_map_scaled
