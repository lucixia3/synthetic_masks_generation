import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    def __init__(self, norm_nc: int, label_nc: int) -> None:
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(128, norm_nc, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.shape[2:], mode="nearest")
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta


class SPADEResnetBlock(nn.Module):
    def __init__(self, fin: int, fout: int, label_nc: int) -> None:
        super().__init__()
        fmiddle = min(fin, fout)
        self.learned_shortcut = fin != fout
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        self.spade_0 = SPADE(fin, label_nc)
        self.spade_1 = SPADE(fmiddle, label_nc)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
            self.spade_s = SPADE(fin, label_nc)

    def forward(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        x_s = self.shortcut(x, segmap)
        dx = self.conv_0(self.spade_0(x, segmap))
        dx = F.relu(dx)
        dx = self.conv_1(self.spade_1(dx, segmap))
        dx = F.relu(dx)
        return x_s + dx

    def shortcut(self, x: torch.Tensor, segmap: torch.Tensor) -> torch.Tensor:
        if self.learned_shortcut:
            return self.conv_s(self.spade_s(x, segmap))
        return x


class SPADEGenerator(nn.Module):
    def __init__(self, label_nc: int, ngf: int = 64, z_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Conv2d(z_dim, ngf * 16, kernel_size=3, padding=1)
        self.head = SPADEResnetBlock(ngf * 16, ngf * 16, label_nc)
        self.G_middle_0 = SPADEResnetBlock(ngf * 16, ngf * 16, label_nc)
        self.G_middle_1 = SPADEResnetBlock(ngf * 16, ngf * 16, label_nc)
        self.up_0 = SPADEResnetBlock(ngf * 16, ngf * 8, label_nc)
        self.up_1 = SPADEResnetBlock(ngf * 8, ngf * 4, label_nc)
        self.up_2 = SPADEResnetBlock(ngf * 4, ngf * 2, label_nc)
        self.up_3 = SPADEResnetBlock(ngf * 2, ngf, label_nc)
        self.conv_img = nn.Conv2d(ngf, 1, kernel_size=3, padding=1)

    def forward(self, segmap: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = self.head(x, segmap)
        x = F.interpolate(x, scale_factor=2)
        x = self.G_middle_0(x, segmap)
        x = self.G_middle_1(x, segmap)
        x = F.interpolate(x, scale_factor=2)
        x = self.up_0(x, segmap)
        x = F.interpolate(x, scale_factor=2)
        x = self.up_1(x, segmap)
        x = F.interpolate(x, scale_factor=2)
        x = self.up_2(x, segmap)
        x = F.interpolate(x, scale_factor=2)
        x = self.up_3(x, segmap)
        x = torch.tanh(self.conv_img(F.relu(x)))
        return x


class PatchDiscriminator(nn.Module):
    def __init__(self, input_nc: int, ndf: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                ),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw),
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
