"""Neural network architectures for EEG/EMG fusion."""

from __future__ import annotations

import math
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_TAP_LABELS: dict[str, str] = {
    "eeg": "EEG only",
    "emg": "EMG only",
    "fused": "Fused (pre-classifier)",
}


def _resolve_modality_flags(
    *,
    use_eeg: bool | None,
    use_emg: bool | None,
    n_eeg: int,
    n_emg: int,
) -> tuple[bool, bool]:
    resolved_use_eeg = (n_eeg > 0) if use_eeg is None else use_eeg
    resolved_use_emg = (n_emg > 0) if use_emg is None else use_emg
    if not resolved_use_eeg and not resolved_use_emg:
        raise ValueError("At least one modality must be enabled")
    if resolved_use_eeg and n_eeg <= 0:
        raise ValueError("n_eeg must be > 0 when EEG is enabled")
    if resolved_use_emg and n_emg <= 0:
        raise ValueError("n_emg must be > 0 when EMG is enabled")
    return resolved_use_eeg, resolved_use_emg


class ModalityBranch(nn.Module):
    """EEGNet Block 1: temporal conv -> depthwise spatial conv.

    Input:  (B, 1, C, T)
    Output: (B, D*F1, 1, T)   -- spatial axis collapsed, time preserved
    """

    def __init__(self, n_channels: int, F1: int, D: int, kernel_length: int):
        super().__init__()
        self.temporal = nn.Conv2d(
            1, F1, (1, kernel_length),
            padding=(0, kernel_length // 2), bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial = nn.Conv2d(
            F1, D * F1, (n_channels, 1),
            groups=F1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(D * F1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.temporal(x))          # (B, F1, C, T)
        x = self.bn2(self.spatial(x))           # (B, D*F1, 1, T)
        x = F.elu(x)
        return x


class TimeAvgPool(nn.Module):
    """Pool along the time axis to a fixed length without AdaptiveAvgPool2d."""

    def __init__(self, out_len: int):
        super().__init__()
        self.out_len = out_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[-1]
        out = self.out_len
        if t == out:
            return x
        if t < out:
            return F.interpolate(x, size=(1, out), mode="linear", align_corners=False)

        trim = t - (t % out)
        x = x[..., :trim]
        stride = trim // out
        return F.avg_pool2d(x, kernel_size=(1, stride), stride=(1, stride))


class IntermediateFusionEEGNet(nn.Module):
    """EEGNet Block-1 branch(es) fused before the separable conv."""

    EMBEDDING_TAPS: ClassVar[dict[str, str]] = EMBEDDING_TAP_LABELS

    def __init__(
        self,
        n_eeg: int,
        n_emg: int,
        n_classes: int,
        T: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 32,
        kern_eeg: int = 128,
        kern_emg: int = 128,
        sep_kernel: int = 16,
        p_drop: float = 0.25,
        *,
        use_eeg: bool | None = None,
        use_emg: bool | None = None,
    ):
        super().__init__()
        self.use_eeg, self.use_emg = _resolve_modality_flags(
            use_eeg=use_eeg,
            use_emg=use_emg,
            n_eeg=n_eeg,
            n_emg=n_emg,
        )
        self.branch_maps = D * F1

        if self.use_eeg:
            self.eeg_branch = ModalityBranch(n_eeg, F1, D, kern_eeg)
        if self.use_emg:
            self.emg_branch = ModalityBranch(n_emg, F1, D, kern_emg)

        fused_maps = self.branch_maps * (int(self.use_eeg) + int(self.use_emg))

        pool1_out = max(1, T // 4)
        pool2_out = max(1, pool1_out // 8)
        self.pool1_out = pool1_out
        self.pool2_out = pool2_out

        self.pool1 = TimeAvgPool(pool1_out)
        self.drop1 = nn.Dropout(p_drop)

        self.sep_depth = nn.Conv2d(
            fused_maps, fused_maps, (1, sep_kernel),
            padding=(0, sep_kernel // 2), groups=fused_maps, bias=False,
        )
        self.sep_point = nn.Conv2d(fused_maps, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = TimeAvgPool(pool2_out)
        self.drop2 = nn.Dropout(p_drop)

        self.classifier = nn.Linear(F2 * pool2_out, n_classes)

    def _encode_branches(
        self, eeg: torch.Tensor, emg: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        branches: list[torch.Tensor] = []
        if self.use_eeg:
            branches.append(self.eeg_branch(eeg))
        if self.use_emg:
            branches.append(self.emg_branch(emg))
        if len(branches) == 2:
            t = min(branches[0].shape[-1], branches[1].shape[-1])
            return branches[0][..., :t], branches[1][..., :t]
        return (branches[0],)

    def _fusion_embed(self, *branch_maps: torch.Tensor, apply_dropout: bool) -> torch.Tensor:
        x = branch_maps[0] if len(branch_maps) == 1 else torch.cat(branch_maps, dim=1)
        x = self.pool1(x)
        if apply_dropout:
            x = self.drop1(x)

        x = self.sep_point(self.sep_depth(x))
        x = F.elu(self.bn3(x))
        x = self.pool2(x)
        if apply_dropout:
            x = self.drop2(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, eeg: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        maps = self._encode_branches(eeg, emg)
        fused = self._fusion_embed(*maps, apply_dropout=True)
        return self.classifier(fused)

    def forward_embeddings(self, eeg: torch.Tensor, emg: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-modality and fused embeddings without dropout (use under model.eval())."""
        if self.use_eeg and self.use_emg:
            e, m = self._encode_branches(eeg, emg)
            zero_e = torch.zeros_like(e)
            zero_m = torch.zeros_like(m)
            return {
                "eeg": self._fusion_embed(e, zero_m, apply_dropout=False),
                "emg": self._fusion_embed(zero_e, m, apply_dropout=False),
                "fused": self._fusion_embed(e, m, apply_dropout=False),
            }

        maps = self._encode_branches(eeg, emg)
        fused = self._fusion_embed(*maps, apply_dropout=False)
        if self.use_eeg:
            return {"eeg": fused}
        return {"emg": fused}


def _tensor_bc_t(x: torch.Tensor) -> torch.Tensor:
    """(B, 1, C, T) -> (B, T, C)."""
    return x.squeeze(1).transpose(1, 2)


def _with_temporal_diff(x: torch.Tensor) -> torch.Tensor:
    """Concatenate each timestep with its first-order temporal difference."""
    if x.shape[1] < 2:
        return torch.cat([x, torch.zeros_like(x)], dim=-1)
    diff = x[:, 1:, :] - x[:, :-1, :]
    return torch.cat([x[:, :-1, :], diff], dim=-1)


class ChannelAttention1d(nn.Module):
    """CBAM-style channel attention for (B, C, T) feature maps."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.max_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=2)
        mx = x.amax(dim=2)
        weights = torch.sigmoid(self.avg_mlp(avg) + self.max_mlp(mx))
        return x * weights.unsqueeze(-1)


class ModalityEncoder(nn.Module):
    """Spatial-temporal encoder for one modality (CAT-Net stage 1)."""

    def __init__(
        self,
        n_channels: int,
        conv_dims: tuple[int, int] = (64, 128),
        lstm_hidden: int = 64,
    ):
        super().__init__()
        in_dim = 2 * n_channels
        c1, c2 = conv_dims
        self.conv1 = nn.Conv1d(in_dim, c1, kernel_size=1, bias=True)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=1, bias=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.channel_attn = ChannelAttention1d(c2)
        self.temporal = nn.LSTM(
            input_size=c2,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.out_dim = 2 * lstm_hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = _with_temporal_diff(_tensor_bc_t(x))
        h = seq.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = self.pool(h)
        h = self.channel_attn(h)
        h = h.transpose(1, 2)
        out, _ = self.temporal(h)
        return out


class CATNet(nn.Module):
    """Cross-attention EEG-EMG fusion network (CAT-Net, without domain adversary)."""

    EMBEDDING_TAPS: ClassVar[dict[str, str]] = EMBEDDING_TAP_LABELS

    def __init__(
        self,
        n_eeg: int,
        n_emg: int,
        n_classes: int,
        T: int,
        *,
        conv_dims: tuple[int, int] = (64, 128),
        lstm_hidden: int = 64,
        attn_heads: int = 4,
        attn_dim: int = 128,
        fusion_dim: int = 128,
        p_drop: float = 0.4,
        use_eeg: bool | None = None,
        use_emg: bool | None = None,
    ):
        super().__init__()
        del T
        self.use_eeg, self.use_emg = _resolve_modality_flags(
            use_eeg=use_eeg,
            use_emg=use_emg,
            n_eeg=n_eeg,
            n_emg=n_emg,
        )
        self.fusion_dim = fusion_dim
        self.drop = nn.Dropout(p_drop)

        if self.use_eeg:
            self.eeg_encoder = ModalityEncoder(n_eeg, conv_dims=conv_dims, lstm_hidden=lstm_hidden)
        if self.use_emg:
            self.emg_encoder = ModalityEncoder(n_emg, conv_dims=conv_dims, lstm_hidden=lstm_hidden)

        if self.use_eeg and self.use_emg:
            embed_dim = self.eeg_encoder.out_dim
            if embed_dim != attn_dim:
                raise ValueError(
                    f"encoder output dim {embed_dim} must match attn_dim {attn_dim}; "
                    "adjust lstm_hidden or attn_dim"
                )
            self.eeg_cross = nn.MultiheadAttention(
                embed_dim, attn_heads, batch_first=True,
            )
            self.emg_cross = nn.MultiheadAttention(
                embed_dim, attn_heads, batch_first=True,
            )
            self.norm_eeg = nn.LayerNorm(embed_dim)
            self.norm_emg = nn.LayerNorm(embed_dim)
            self.proj_eeg = nn.Linear(embed_dim * 2, fusion_dim)
            self.proj_emg = nn.Linear(embed_dim * 2, fusion_dim)
            self.classifier = nn.Linear(fusion_dim * 2, n_classes)
        elif self.use_eeg:
            self.proj_eeg = nn.Linear(self.eeg_encoder.out_dim * 2, fusion_dim)
            self.classifier = nn.Linear(fusion_dim, n_classes)
        else:
            self.proj_emg = nn.Linear(self.emg_encoder.out_dim * 2, fusion_dim)
            self.classifier = nn.Linear(fusion_dim, n_classes)

    def _pool_proj(self, z: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        pooled = torch.cat([z.mean(dim=1), z.amax(dim=1)], dim=-1)
        return proj(pooled)

    def _encode_pair(
        self, eeg: torch.Tensor, emg: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        z_eeg = self.eeg_encoder(eeg) if self.use_eeg else None
        z_emg = self.emg_encoder(emg) if self.use_emg else None
        if z_eeg is not None and z_emg is not None:
            t = min(z_eeg.shape[1], z_emg.shape[1])
            return z_eeg[:, :t, :], z_emg[:, :t, :]
        return z_eeg, z_emg

    def _cross_fuse(
        self, z_eeg: torch.Tensor, z_emg: torch.Tensor, *, apply_dropout: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c_eeg, _ = self.eeg_cross(z_eeg, z_emg, z_emg)
        c_emg, _ = self.emg_cross(z_emg, z_eeg, z_eeg)
        c_eeg = self.norm_eeg(c_eeg)
        c_emg = self.norm_emg(c_emg)

        f_eeg = self._pool_proj(c_eeg, self.proj_eeg)
        f_emg = self._pool_proj(c_emg, self.proj_emg)
        fused = torch.cat([f_eeg, f_emg], dim=-1)
        if apply_dropout:
            fused = self.drop(fused)
        return f_eeg, f_emg, fused

    def forward(self, eeg: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        z_eeg, z_emg = self._encode_pair(eeg, emg)
        if self.use_eeg and self.use_emg:
            _, _, fused = self._cross_fuse(z_eeg, z_emg, apply_dropout=True)
            return self.classifier(fused)
        if self.use_eeg:
            fused = self._pool_proj(z_eeg, self.proj_eeg)
        else:
            fused = self._pool_proj(z_emg, self.proj_emg)
        if self.drop.p > 0.0:
            fused = self.drop(fused)
        return self.classifier(fused)

    def forward_embeddings(self, eeg: torch.Tensor, emg: torch.Tensor) -> dict[str, torch.Tensor]:
        z_eeg, z_emg = self._encode_pair(eeg, emg)
        if self.use_eeg and self.use_emg:
            f_eeg, f_emg, fused = self._cross_fuse(z_eeg, z_emg, apply_dropout=False)
            return {"eeg": f_eeg, "emg": f_emg, "fused": fused}
        if self.use_eeg:
            return {"eeg": self._pool_proj(z_eeg, self.proj_eeg)}
        return {"emg": self._pool_proj(z_emg, self.proj_emg)}


class CausalConv1d(nn.Module):
    """Left-padded 1D convolution for causal temporal modeling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class ATCNetConvBlock(nn.Module):
    """ATCNet convolutional (CV) block: temporal -> depthwise spatial -> temporal."""

    def __init__(
        self,
        n_channels: int,
        T: int,
        *,
        F1: int = 16,
        D: int = 2,
        kern_temporal: int = 64,
        kern_refine: int = 16,
        pool_size_1: int = 8,
        pool_size_2: int = 7,
        p_drop: float = 0.3,
    ):
        super().__init__()
        self.F2 = F1 * D
        self.Tc = max(1, max(1, T // pool_size_1) // pool_size_2)

        self.conv1 = nn.Conv2d(
            1, F1, (1, kern_temporal),
            padding=(0, kern_temporal // 2), bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(
            F1, self.F2, (n_channels, 1),
            groups=F1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.F2)
        self.pool1 = TimeAvgPool(max(1, T // pool_size_1))
        self.drop1 = nn.Dropout2d(p_drop)

        self.conv3 = nn.Conv2d(
            self.F2, self.F2, (1, kern_refine),
            padding=(0, kern_refine // 2), bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.pool2 = TimeAvgPool(self.Tc)
        self.drop2 = nn.Dropout2d(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.drop1(self.pool1(x))
        x = F.elu(self.bn3(self.conv3(x)))
        x = self.drop2(self.pool2(x))
        return x.squeeze(2)


class ATCNetMHA(nn.Module):
    """Multi-head attention with per-head dim independent of input dim (ATCNet paper)."""

    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        output_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.embed_dim = head_dim * num_heads
        self.fc_q = nn.Linear(input_dim, self.embed_dim)
        self.fc_k = nn.Linear(input_dim, self.embed_dim)
        self.fc_v = nn.Linear(input_dim, self.embed_dim)
        self.fc_o = nn.Linear(self.embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size = q.shape[0]
        q = self.fc_q(q)
        k = self.fc_k(k)
        v = self.fc_v(v)

        q_ = torch.cat(q.split(self.head_dim, dim=-1), dim=0)
        k_ = torch.cat(k.split(self.head_dim, dim=-1), dim=0)
        v_ = torch.cat(v.split(self.head_dim, dim=-1), dim=0)

        weights = torch.softmax(
            q_.bmm(k_.transpose(-2, -1)) / math.sqrt(self.head_dim),
            dim=-1,
        )
        heads = torch.cat(
            weights.bmm(v_).split(batch_size, dim=0),
            dim=-1,
        )
        return self.dropout(self.fc_o(heads))


class ATCNetAttentionBlock(nn.Module):
    """Multi-head self-attention over the temporal sequence (AT block)."""

    def __init__(
        self,
        in_dim: int,
        *,
        head_dim: int = 8,
        num_heads: int = 2,
        p_drop: float = 0.5,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)
        self.mha = ATCNetMHA(
            input_dim=in_dim,
            head_dim=head_dim,
            output_dim=in_dim,
            num_heads=num_heads,
            dropout=p_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, T, C) -> residual add -> (B, C, T)
        seq = x.transpose(1, 2)
        out = self.mha(self.norm(seq), self.norm(seq), self.norm(seq))
        return (seq + out).transpose(1, 2)


class ATCNetTCNResidualBlock(nn.Module):
    """Dilated causal residual block from the ATCNet TC block."""

    def __init__(
        self,
        in_channels: int,
        n_filters: int,
        kernel_size: int,
        dilation: int,
        p_drop: float,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, n_filters, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.conv2 = CausalConv1d(n_filters, n_filters, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.drop = nn.Dropout(p_drop)
        self.skip = (
            nn.Conv1d(in_channels, n_filters, kernel_size=1)
            if in_channels != n_filters
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(F.elu(self.bn1(self.conv1(x))))
        out = self.drop(F.elu(self.bn2(self.conv2(out))))
        return F.elu(out + self.skip(x))


class ATCNet(nn.Module):
    """ATCNet (Altaheri et al., 2022) without sliding windows — raw trial in, logits out.

    CV block encodes spatio-temporal features, multi-head self-attention highlights
    salient time steps, and a dilated causal TCN reads out the last temporal position.
    """

    EMBEDDING_TAPS: ClassVar[dict[str, str]] = EMBEDDING_TAP_LABELS

    def __init__(
        self,
        n_eeg: int,
        n_emg: int,
        n_classes: int,
        T: int,
        *,
        F1: int = 16,
        D: int = 2,
        kern_temporal: int | None = None,
        kern_refine: int = 16,
        pool_size_1: int = 8,
        pool_size_2: int = 7,
        p_drop_cv: float = 0.3,
        head_dim: int = 8,
        num_heads: int = 2,
        p_drop_att: float = 0.5,
        tcn_depth: int = 2,
        tcn_kernel: int = 4,
        p_drop_tcn: float = 0.3,
        use_eeg: bool | None = None,
        use_emg: bool | None = None,
    ):
        super().__init__()
        self.use_eeg, self.use_emg = _resolve_modality_flags(
            use_eeg=use_eeg,
            use_emg=use_emg,
            n_eeg=n_eeg,
            n_emg=n_emg,
        )
        if kern_temporal is None:
            kern_temporal = max(1, min(64, T // 4))

        cv_kwargs = dict(
            T=T,
            F1=F1,
            D=D,
            kern_temporal=kern_temporal,
            kern_refine=kern_refine,
            pool_size_1=pool_size_1,
            pool_size_2=pool_size_2,
            p_drop=p_drop_cv,
        )
        if self.use_eeg:
            self.eeg_cv = ATCNetConvBlock(n_eeg, **cv_kwargs)
        if self.use_emg:
            self.emg_cv = ATCNetConvBlock(n_emg, **cv_kwargs)

        self.F2 = F1 * D
        self.tcn_filters = self.F2
        self.seq_dim = self.F2 * (int(self.use_eeg) + int(self.use_emg))

        self.attention = ATCNetAttentionBlock(
            self.seq_dim,
            head_dim=head_dim,
            num_heads=num_heads,
            p_drop=p_drop_att,
        )
        self.tcn = nn.Sequential(
            *[
                ATCNetTCNResidualBlock(
                    in_channels=self.seq_dim if i == 0 else self.tcn_filters,
                    n_filters=self.tcn_filters,
                    kernel_size=tcn_kernel,
                    dilation=2**i,
                    p_drop=p_drop_tcn,
                )
                for i in range(tcn_depth)
            ]
        )
        self.classifier = nn.Linear(self.tcn_filters, n_classes)

    def _encode_cv(self, eeg: torch.Tensor, emg: torch.Tensor) -> list[torch.Tensor]:
        maps: list[torch.Tensor] = []
        if self.use_eeg:
            maps.append(self.eeg_cv(eeg))
        if self.use_emg:
            maps.append(self.emg_cv(emg))
        if len(maps) == 2:
            t = min(maps[0].shape[-1], maps[1].shape[-1])
            return [m[..., :t] for m in maps]
        return maps

    def _concat_maps(self, maps: list[torch.Tensor]) -> torch.Tensor:
        return maps[0] if len(maps) == 1 else torch.cat(maps, dim=1)

    def _sequence_embed(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.attention(seq)
        x = self.tcn(x)
        return x[..., -1]

    def forward(self, eeg: torch.Tensor, emg: torch.Tensor) -> torch.Tensor:
        seq = self._concat_maps(self._encode_cv(eeg, emg))
        return self.classifier(self._sequence_embed(seq))

    def forward_embeddings(self, eeg: torch.Tensor, emg: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.use_eeg and self.use_emg:
            e_map, m_map = self._encode_cv(eeg, emg)
            zero_e = torch.zeros_like(e_map)
            zero_m = torch.zeros_like(m_map)
            return {
                "eeg": self._sequence_embed(self._concat_maps([e_map, zero_m])),
                "emg": self._sequence_embed(self._concat_maps([zero_e, m_map])),
                "fused": self._sequence_embed(self._concat_maps([e_map, m_map])),
            }

        seq = self._concat_maps(self._encode_cv(eeg, emg))
        embed = self._sequence_embed(seq)
        if self.use_eeg:
            return {"eeg": embed}
        return {"emg": embed}


ARCHITECTURES: dict[str, type[nn.Module]] = {
    "intermediate_fusion_eegnet": IntermediateFusionEEGNet,
    "cat_net": CATNet,
    "atc_net": ATCNet,
}


def get_embedding_taps(model: nn.Module) -> dict[str, str]:
    """Return tap key -> plot title for the modalities present in the model."""
    use_eeg = getattr(model, "use_eeg", True)
    use_emg = getattr(model, "use_emg", True)
    if use_eeg and use_emg:
        return dict(EMBEDDING_TAP_LABELS)
    if use_eeg:
        return {"eeg": EMBEDDING_TAP_LABELS["eeg"]}
    if use_emg:
        return {"emg": EMBEDDING_TAP_LABELS["emg"]}
    raise TypeError(f"{type(model).__name__} has no enabled modalities")


def build_fusion_model(
    architecture: str,
    *,
    n_eeg: int,
    n_emg: int,
    n_classes: int,
    T: int,
    state_dict: dict[str, torch.Tensor] | None = None,
    use_eeg: bool | None = None,
    use_emg: bool | None = None,
    **kwargs,
) -> nn.Module:
    if architecture not in ARCHITECTURES:
        known = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(f"unknown architecture {architecture!r}; expected one of: {known}")

    model_kwargs = dict(kwargs)
    if architecture == "intermediate_fusion_eegnet" and state_dict is not None:
        if "F2" not in model_kwargs and "bn3.weight" in state_dict:
            model_kwargs["F2"] = int(state_dict["bn3.weight"].shape[0])

    model_cls = ARCHITECTURES[architecture]
    return model_cls(
        n_eeg=n_eeg,
        n_emg=n_emg,
        n_classes=n_classes,
        T=T,
        use_eeg=use_eeg,
        use_emg=use_emg,
        **model_kwargs,
    )


__all__ = [
    "ARCHITECTURES",
    "ATCNet",
    "ATCNetAttentionBlock",
    "ATCNetConvBlock",
    "CATNet",
    "CausalConv1d",
    "ChannelAttention1d",
    "EMBEDDING_TAP_LABELS",
    "IntermediateFusionEEGNet",
    "ModalityBranch",
    "ModalityEncoder",
    "TimeAvgPool",
    "build_fusion_model",
    "get_embedding_taps",
]
