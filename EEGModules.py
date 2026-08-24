import torch
import torch.nn as nn


class DepthwiseConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        if out_channels % in_channels != 0:
            raise ValueError("out_channels must be divisible by in_channels for depthwise convolution.")
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )


def _clip_row_norms(parameter, max_norm):
    if parameter is None or max_norm is None:
        return
    with torch.no_grad():
        flat = parameter.data.view(parameter.data.shape[0], -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.clamp(float(max_norm) / norms, max=1.0)
        parameter.data.mul_(scale.view(-1, *([1] * (parameter.data.dim() - 1))))


class ConstraintDepthwiseConv2d(DepthwiseConv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, input):
        _clip_row_norms(self.weight, self.max_norm)
        return super().forward(input)


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        depth_multiplier=1,
        bias=False,
    ):
        super().__init__()
        hidden_channels = in_channels * depth_multiplier
        self.depthwise = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ConstraintLazyLinear(nn.LazyLinear):
    def __init__(self, out_features, bias=True, device=None, dtype=None, max_norm=0.25):
        super().__init__(out_features, bias=bias, device=device, dtype=dtype)
        self.max_norm = max_norm

    def forward(self, input):
        if not self.has_uninitialized_params():
            _clip_row_norms(self.weight, self.max_norm)
        return super().forward(input)
