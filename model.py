import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import model as base

IGNORE_INDEX = base.IGNORE_INDEX

_normalize_index = base._normalize_index
sampler = base.sampler
CDDataSet = base.CDDataSet
CriterionMask = base.CriterionMask
dice_binary_from_multiclass = base.dice_binary_from_multiclass
confusion_guided_fp_margin_loss = base.confusion_guided_fp_margin_loss


def _center_crop_like(logits, target):
    return base._center_crop_like(logits, target)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, groups=1, dropout=0.0):
        super().__init__()
        pad = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class ResidualDWBlock(nn.Module):
    def __init__(self, ch, dilation=1, dropout=0.0):
        super().__init__()
        self.body = nn.Sequential(
            ConvBNAct(ch, ch, kernel_size=3, dilation=dilation, groups=ch, dropout=dropout),
            nn.Conv2d(ch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.body(x))


class SSDScan1D(nn.Module):
    def __init__(self, dim, rank=8, dropout=0.05, bidirectional=True):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(max(1, min(rank, dim)))
        self.bidirectional = bool(bidirectional)

        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, self.rank * 3 + dim, bias=True)
        self.out = nn.Linear(self.rank, dim, bias=True)
        self.drop = nn.Dropout(dropout)

    def _scan(self, z):
        q, k, v, gate = self.proj(z).split([self.rank, self.rank, self.rank, self.dim], dim=-1)
        q = torch.tanh(q) / math.sqrt(max(self.rank, 1))
        k = torch.tanh(k)
        v = F.silu(v)

        state = torch.cumsum(k * v, dim=1)
        denom = torch.cumsum(k.abs(), dim=1).clamp_min(1.0)
        y = q * state / denom
        return y, torch.sigmoid(gate)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"SSDScan1D expects [B,L,C], got {tuple(x.shape)}")
        z = self.norm(x)
        yf, gate = self._scan(z)
        if self.bidirectional:
            yb, _ = self._scan(torch.flip(z, dims=[1]))
            yb = torch.flip(yb, dims=[1])
            y = 0.5 * (yf + yb)
        else:
            y = yf
        return x + self.drop(self.out(y) * gate)


class SpectralSequenceMamba(nn.Module):
    def __init__(self, out_ch, spectral_tokens=24, token_dim=16, rank=8, dropout=0.05):
        super().__init__()
        self.spectral_tokens = int(spectral_tokens)
        self.token_dim = int(token_dim)
        self.embed = nn.Linear(1, token_dim)
        self.scan1 = SSDScan1D(token_dim, rank=min(rank, token_dim), dropout=dropout)
        self.scan2 = SSDScan1D(token_dim, rank=min(rank, token_dim), dropout=dropout)
        self.out = nn.Sequential(
            nn.Linear(token_dim * 2, out_ch),
            nn.LayerNorm(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        spec = x.permute(0, 2, 3, 1).reshape(b * h * w, 1, c)
        spec = F.adaptive_avg_pool1d(spec, self.spectral_tokens)
        spec = spec.transpose(1, 2).contiguous()

        z = self.embed(spec)
        z = self.scan2(self.scan1(z))
        pooled = torch.cat([z.mean(dim=1), z.amax(dim=1)], dim=-1)
        out = self.out(pooled)
        return out.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()


class SpatialMambaBlock(nn.Module):
    def __init__(self, ch, rank=8, dropout=0.05, scan_mode="rowcol"):
        super().__init__()
        if scan_mode not in ["rowcol", "row", "col"]:
            raise ValueError(f"Unsupported scan_mode: {scan_mode}")
        self.scan_mode = scan_mode
        self.local = ResidualDWBlock(ch, dilation=1, dropout=dropout)
        self.row = SSDScan1D(ch, rank=rank, dropout=dropout)
        self.col = SSDScan1D(ch, rank=rank, dropout=dropout)
        n_branch = 2 if scan_mode == "rowcol" else 1
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * n_branch, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU(),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(ch, ch * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()

    def _row_scan(self, x):
        b, c, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(b * h, w, c).contiguous()
        out = self.row(seq)
        return out.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    def _col_scan(self, x):
        b, c, h, w = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(b * w, h, c).contiguous()
        out = self.col(seq)
        return out.reshape(b, w, h, c).permute(0, 3, 2, 1).contiguous()

    def forward(self, x):
        x = self.local(x)
        outs = []
        if self.scan_mode in ["rowcol", "row"]:
            outs.append(self._row_scan(x))
        if self.scan_mode in ["rowcol", "col"]:
            outs.append(self._col_scan(x))
        y = self.fuse(torch.cat(outs, dim=1))
        x = self.act(x + y)
        return self.act(x + self.ffn(x))


class SpatialBranch(nn.Module):
    def __init__(self, in_ch, out_ch, depth=3, rank=8, dropout=0.05, scan_mode="rowcol"):
        super().__init__()
        layers = [
            ConvBNAct(in_ch, out_ch, kernel_size=1, dropout=dropout),
            ResidualDWBlock(out_ch, dilation=1, dropout=dropout),
        ]
        for _ in range(max(1, depth)):
            layers.append(SpatialMambaBlock(out_ch, rank=rank, dropout=dropout, scan_mode=scan_mode))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TemporalInteractionMamba(nn.Module):
    def __init__(self, ch, rank=8, dropout=0.05):
        super().__init__()
        self.scan = SSDScan1D(ch, rank=rank, dropout=dropout)
        self.refine = nn.Sequential(
            ConvBNAct(ch * 3 + 1, ch, kernel_size=1, dropout=dropout),
            ResidualDWBlock(ch, dilation=1, dropout=dropout),
            ResidualDWBlock(ch, dilation=2, dropout=dropout),
        )

    @staticmethod
    def angle_proxy(f1, f2, eps=1e-6):
        dot = (f1 * f2).sum(dim=1, keepdim=True)
        n1 = torch.sqrt((f1 * f1).sum(dim=1, keepdim=True).clamp_min(eps))
        n2 = torch.sqrt((f2 * f2).sum(dim=1, keepdim=True).clamp_min(eps))
        return 1.0 - (dot / (n1 * n2 + eps)).clamp(-1.0, 1.0)

    def forward(self, f1, f2):
        b, c, h, w = f1.shape
        seq = torch.stack([f1, f2], dim=2)
        seq = seq.permute(0, 3, 4, 2, 1).reshape(b * h * w, 2, c).contiguous()
        z = self.scan(seq)
        z = z.reshape(b, h, w, 2, c).permute(0, 4, 3, 1, 2).contiguous()
        t1, t2 = z[:, :, 0], z[:, :, 1]
        ds = t2 - t1
        da = ds.abs()
        ang = self.angle_proxy(t1, t2)
        return self.refine(torch.cat([ds, da, t2, ang], dim=1))

class AdaptiveResidualFusion(nn.Module):
    def __init__(self, ch, branches=3, dropout=0.05):
        super().__init__()
        self.branches = int(branches)
        self.base = nn.Sequential(
            nn.Conv2d(ch * self.branches, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU(),
        )
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False),
                nn.BatchNorm2d(ch),
                nn.GELU(),
                nn.Conv2d(ch, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            for _ in range(self.branches)
        ])
        self.alpha = nn.Parameter(torch.full((self.branches,), 0.1))
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, features):
        if len(features) != self.branches:
            raise ValueError(f"Expected {self.branches} branches, got {len(features)}")
        fused = self.base(torch.cat(features, dim=1))
        for i, (gate, feat) in enumerate(zip(self.gates, features)):
            fused = fused + self.alpha[i] * gate(feat) * feat
        return self.dropout(fused)


class MulticlassHead(nn.Module):
    def __init__(self, ch, num_classes, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            ResidualDWBlock(ch, dropout=dropout),
            ConvBNAct(ch, ch, kernel_size=1, dropout=dropout),
            nn.Conv2d(ch, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class CD(nn.Module):
    def __init__(
        self,
        num_classes,
        in_ch=None,
        spectral_tokens=24,
        d_model=64,
        feat_ch=64,
        rank=8,
        depth=3,
        dropout=0.05,
        diff_mode="full",
        scan_mode="rowcol",
        fusion_mode="residual",
        drop_spectral=False,
        drop_spatial=False,
        drop_temporal=False,
    ):
        super().__init__()
        if in_ch is None:
            raise ValueError("model_sst_gdamamba.CD requires in_ch")

        if fusion_mode not in ["softmax", "sum", "residual"]:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        self.fusion_mode = fusion_mode
        self.drop_spectral = bool(drop_spectral)
        self.drop_spatial = bool(drop_spatial)
        self.drop_temporal = bool(drop_temporal)
        self.d_model = int(d_model)

        token_dim = max(8, min(24, d_model // 4))
        self.spectral = SpectralSequenceMamba(
            out_ch=d_model,
            spectral_tokens=spectral_tokens,
            token_dim=token_dim,
            rank=rank,
            dropout=dropout,
        )

        spatial_in = 0
        if diff_mode != "no_ds":
            spatial_in += in_ch
        if diff_mode != "no_da":
            spatial_in += in_ch
        if diff_mode != "no_angle":
            spatial_in += 1
        if spatial_in == 0:
            spatial_in = in_ch
        self.diff_mode = diff_mode
        self.spatial = SpatialBranch(
            in_ch=spatial_in,
            out_ch=d_model,
            depth=depth,
            rank=rank,
            dropout=dropout,
            scan_mode=scan_mode,
        )

        self.shared_embed = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.GELU(),
            ResidualDWBlock(feat_ch, dilation=1, dropout=dropout),
            nn.Conv2d(feat_ch, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )
        self.temporal = TemporalInteractionMamba(d_model, rank=rank, dropout=dropout)
        if fusion_mode == "softmax":
            self.fusion = SoftmaxGatedFusion(d_model, branches=3, dropout=dropout)
        elif fusion_mode == "sum":
            self.fusion = SumFusion(d_model, dropout=dropout)
        elif fusion_mode == "residual":
            self.fusion = AdaptiveResidualFusion(d_model, branches=3, dropout=dropout)
        self.head = BinaryGuidedHead(d_model, num_classes, dropout=dropout, guide_scale=0.5)

    @staticmethod
    def angle_proxy(x1, x2, eps=1e-6):
        dot = (x1 * x2).sum(dim=1, keepdim=True)
        n1 = torch.sqrt((x1 * x1).sum(dim=1, keepdim=True).clamp_min(eps))
        n2 = torch.sqrt((x2 * x2).sum(dim=1, keepdim=True).clamp_min(eps))
        return 1.0 - (dot / (n1 * n2 + eps)).clamp(-1.0, 1.0)

    def spatial_parts(self, x1, x2):
        ds = x2 - x1
        parts = []
        if self.diff_mode != "no_ds":
            parts.append(ds)
        if self.diff_mode != "no_da":
            parts.append(ds.abs())
        if self.diff_mode != "no_angle":
            parts.append(self.angle_proxy(x1, x2))
        if not parts:
            parts.append(ds.abs())
        return torch.cat(parts, dim=1)

    def forward(self, x, target=None, update_proto=False):
        if x.ndim != 5:
            raise ValueError(f"Expected input [B,2,C,H,W], got {tuple(x.shape)}")

        x1 = x[:, 0]
        x2 = x[:, 1]

        diff = x2 - x1
        spatial_input = self.spatial_parts(x1, x2)
        zero = x1.new_zeros(x1.shape[0], self.d_model, x1.shape[2], x1.shape[3])

        spec = zero if self.drop_spectral else self.spectral(diff)
        spa = zero if self.drop_spatial else self.spatial(spatial_input)
        f1 = self.shared_embed(x1)
        f2 = self.shared_embed(x2)
        tmp = zero if self.drop_temporal else self.temporal(f1, f2)

        z = self.fusion([spec, spa, tmp])
        return self.head(z)


def CriterionPatch(
    logits,
    target,
    ignore_index=IGNORE_INDEX,
    pair_matrix=None,
    fp_margin_weight=0.0,
    fp_margin_value=0.35,
    class_weights=None,
    focal_gamma=1.5,
    binary_dice_weight=0.25,
):
    return base.CriterionPatch(
        logits,
        target,
        ignore_index=ignore_index,
        pair_matrix=pair_matrix,
        fp_margin_weight=fp_margin_weight,
        fp_margin_value=fp_margin_value,
        class_weights=class_weights,
        focal_gamma=focal_gamma,
        binary_dice_weight=binary_dice_weight,
    )
