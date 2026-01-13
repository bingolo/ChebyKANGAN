import torch
import torch.nn as nn

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.coeffs = nn.Parameter(torch.randn(output_dim, input_dim, degree + 1) * 0.001)

    def forward(self, x):
        x = torch.clamp(x, -10, 10)
        x = torch.tanh(x).unsqueeze(-1)
        cheby = [torch.ones_like(x), x]
        for _ in range(2, self.degree + 1):
            nxt = 2 * x * cheby[-1] - cheby[-2]
            nxt = torch.clamp(nxt, -10, 10)
            cheby.append(nxt)
        cheby = torch.cat(cheby, dim=-1)  # [N, in_dim, degree+1]
        out = torch.einsum("nid,oid->no", cheby, self.coeffs)
        return torch.clamp(out, -10, 10)

class ChebyKANConv2d(nn.Module):
    def __init__(self, channels, degree=4):
        super().__init__()
        self.ck = ChebyKANLayer(channels, channels, degree=degree)
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(-1, C)
        flat = self.ck(flat)
        flat = self.ln(flat)
        return flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        mid = max(8, ch // 8)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1),
            nn.ReLU(True),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid(),
        )
        self.sa = nn.Sequential(nn.Conv2d(ch, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch)
        )

    def forward(self, x):
        return x + self.block(x)
