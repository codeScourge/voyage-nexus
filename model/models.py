"""Neural network architectures for EEG/EMG fusion."""

from __future__ import annotations

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


ARCHITECTURES: dict[str, type[nn.Module]] = {
    "intermediate_fusion_eegnet": IntermediateFusionEEGNet,
    "cat_net": CATNet,
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
    "CATNet",
    "ChannelAttention1d",
    "EMBEDDING_TAP_LABELS",
    "IntermediateFusionEEGNet",
    "ModalityBranch",
    "ModalityEncoder",
    "TimeAvgPool",
    "build_fusion_model",
    "get_embedding_taps",
]
