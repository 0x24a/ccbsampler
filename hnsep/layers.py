from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def crop_center(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    if h1.size(3) == h2.size(3):
        return h1
    if h1.size(3) < h2.size(3):
        raise ValueError("h1 time dimension must be >= h2")
    s = (h1.size(3) - h2.size(3)) // 2
    return h1[:, :, :, s : s + h2.size(3)]


class Conv2DBNActiv(nn.Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        ksize: int = 3,
        stride: int = 1,
        pad: int = 1,
        dilation: int = 1,
        activ: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nin, nout, kernel_size=ksize, stride=stride,
                      padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(nout),
            activ(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # type: ignore[return-value]


class Encoder(nn.Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        ksize: int = 3,
        stride: int = 1,
        pad: int = 1,
        activ: type[nn.Module] = nn.LeakyReLU,
    ) -> None:
        super().__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, stride, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class Decoder(nn.Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        ksize: int = 3,
        stride: int = 1,
        pad: int = 1,
        activ: type[nn.Module] = nn.ReLU,
        dropout: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
        fixed_length: bool = True,
    ) -> torch.Tensor:
        if fixed_length:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        else:
            _, _, h, w = x.size()
            x = F.pad(x, (0, 1, 0, 1), mode="replicate")
            x = F.interpolate(x, size=(2 * h + 1, 2 * w + 1), mode="bilinear", align_corners=True)
            x = x[:, :, :-1, :-1]

        if skip is not None:
            x = torch.cat([crop_center(skip, x), x], dim=1)

        h_out = self.conv1(x)
        if self.dropout is not None:
            h_out = self.dropout(h_out)
        return h_out


class Mean(nn.Module):
    def __init__(self, dim: int, keepdims: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.keepdims = keepdims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(self.dim, keepdim=self.keepdims)


class ASPPModule(nn.Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        dilations: tuple[int, ...] = (4, 8, 12),
        activ: type[nn.Module] = nn.ReLU,
        dropout: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(Mean(dim=-2, keepdims=True),
                                   Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ))
        self.conv2 = Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ)
        self.conv3 = Conv2DBNActiv(nin, nout, 3, 1, dilations[0], dilations[0], activ=activ)
        self.conv4 = Conv2DBNActiv(nin, nout, 3, 1, dilations[1], dilations[1], activ=activ)
        self.conv5 = Conv2DBNActiv(nin, nout, 3, 1, dilations[2], dilations[2], activ=activ)
        self.bottleneck = Conv2DBNActiv(nout * 5, nout, 1, 1, 0, activ=activ)
        self.dropout = nn.Dropout2d(0.1) if dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, _ = x.size()
        feat1 = self.conv1(x).repeat(1, 1, h, 1)
        out = torch.cat((feat1, self.conv2(x), self.conv3(x), self.conv4(x), self.conv5(x)), dim=1)
        out = self.bottleneck(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class LSTMModule(nn.Module):
    def __init__(self, nin_conv: int, nin_lstm: int, nout_lstm: int) -> None:
        super().__init__()
        self.conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0)
        self.lstm = nn.LSTM(input_size=nin_lstm, hidden_size=nout_lstm // 2, bidirectional=True)
        self.dense = nn.Sequential(
            nn.Linear(nout_lstm, nin_lstm),
            nn.BatchNorm1d(nin_lstm),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, _, nbins, nframes = x.size()
        h = self.conv(x)[:, 0].permute(2, 0, 1)  # nframes, N, nbins
        h, _ = self.lstm(h)
        h = self.dense(h.reshape(-1, h.size(-1)))
        return h.reshape(nframes, N, 1, nbins).permute(1, 2, 3, 0)
