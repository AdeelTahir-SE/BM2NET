"""
BM2-Net model architecture.

Architecture (from diagram):
    Input tensor (3×640×960)
        → MNASNet backbone (depth multiplier 1)
        → Split + project to tokens + position & view embeddings
        → Cross-view attention (bidirectional):
            - Frontal ← Lateral (queries from frontal view)
            - Lateral ← Frontal (queries from lateral view)
        → Concat + pool (residual, layer norm, fused vector)
        → MLP head (128 hidden units)
        → 14 body measurements
"""

import torch
import torch.nn as nn
import torchvision.models as models


class TokenProjector(nn.Module):
    """
    Split backbone features into frontal/lateral halves, flatten to token
    sequences, project to d_model, and add positional + view embeddings.
    """

    def __init__(self, in_channels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Linear projection from backbone channels to d_model
        self.proj = nn.Linear(in_channels, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable view embeddings (frontal=0, lateral=1)
        self.view_embedding = nn.Embedding(2, d_model)

        # Position embeddings are lazily initialized based on spatial dims
        self._pos_embedding = None
        self._num_tokens = None

    def _init_pos_embedding(self, num_tokens: int, device: torch.device):
        """Lazily create position embeddings once we know the spatial size."""
        if self._num_tokens != num_tokens:
            self._num_tokens = num_tokens
            self._pos_embedding = nn.Parameter(
                torch.randn(1, num_tokens, self.d_model, device=device) * 0.02
            )

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: Backbone output, shape (B, C, H, W)
                      where W is the full width (frontal + lateral side-by-side).

        Returns:
            frontal_tokens: (B, num_tokens, d_model)
            lateral_tokens: (B, num_tokens, d_model)
        """
        B, C, H, W = features.shape
        W_half = W // 2

        # Split into frontal (left half) and lateral (right half)
        frontal_feat = features[:, :, :, :W_half]   # (B, C, H, W/2)
        lateral_feat = features[:, :, :, W_half:]    # (B, C, H, W/2)

        # Flatten spatial dims → token sequences: (B, C, H, W/2) → (B, H*W/2, C)
        frontal_tokens = frontal_feat.flatten(2).transpose(1, 2)  # (B, N, C)
        lateral_tokens = lateral_feat.flatten(2).transpose(1, 2)  # (B, N, C)

        num_tokens = frontal_tokens.shape[1]

        # Project to d_model
        frontal_tokens = self.proj(frontal_tokens)  # (B, N, d_model)
        lateral_tokens = self.proj(lateral_tokens)  # (B, N, d_model)

        # Add position embeddings (lazy init)
        self._init_pos_embedding(num_tokens, features.device)
        frontal_tokens = frontal_tokens + self._pos_embedding
        lateral_tokens = lateral_tokens + self._pos_embedding

        # Add view embeddings
        frontal_view = self.view_embedding(
            torch.zeros(B, num_tokens, dtype=torch.long, device=features.device)
        )
        lateral_view = self.view_embedding(
            torch.ones(B, num_tokens, dtype=torch.long, device=features.device)
        )
        frontal_tokens = frontal_tokens + frontal_view
        lateral_tokens = lateral_tokens + lateral_view

        # Norm + dropout
        frontal_tokens = self.dropout(self.norm(frontal_tokens))
        lateral_tokens = self.dropout(self.norm(lateral_tokens))

        return frontal_tokens, lateral_tokens


class CrossViewAttention(nn.Module):
    """
    Bidirectional cross-view attention:
        - Frontal ← Lateral: frontal tokens as queries, lateral as keys/values
        - Lateral ← Frontal: lateral tokens as queries, frontal as keys/values
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        # Frontal ← Lateral
        self.cross_attn_f2l = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_f = nn.LayerNorm(d_model)

        # Lateral ← Frontal
        self.cross_attn_l2f = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_l = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        frontal_tokens: torch.Tensor,
        lateral_tokens: torch.Tensor,
    ):
        """
        Args:
            frontal_tokens: (B, N, d_model)
            lateral_tokens: (B, N, d_model)

        Returns:
            frontal_updated: (B, N, d_model) with residual
            lateral_updated: (B, N, d_model) with residual
        """
        # Frontal ← Lateral: Q=frontal, K=V=lateral
        attn_f, _ = self.cross_attn_f2l(
            query=frontal_tokens,
            key=lateral_tokens,
            value=lateral_tokens,
        )
        frontal_updated = self.norm_f(frontal_tokens + self.dropout(attn_f))

        # Lateral ← Frontal: Q=lateral, K=V=frontal
        attn_l, _ = self.cross_attn_l2f(
            query=lateral_tokens,
            key=frontal_tokens,
            value=frontal_tokens,
        )
        lateral_updated = self.norm_l(lateral_tokens + self.dropout(attn_l))

        return frontal_updated, lateral_updated


class FusionBlock(nn.Module):
    """
    Concat + pool: concatenate cross-attended view tokens,
    apply global average pooling, residual connection, and layer norm
    to produce a fixed-length fused vector.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # After concatenating frontal + lateral pooled vectors → 2*d_model
        self.fused_dim = d_model * 2
        self.layer_norm = nn.LayerNorm(self.fused_dim)
        self.residual_proj = nn.Linear(self.fused_dim, self.fused_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        frontal_tokens: torch.Tensor,
        lateral_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            frontal_tokens: (B, N, d_model)
            lateral_tokens: (B, N, d_model)

        Returns:
            fused: (B, fused_dim) where fused_dim = 2 * d_model
        """
        # Global average pool over token dimension
        frontal_pooled = frontal_tokens.mean(dim=1)  # (B, d_model)
        lateral_pooled = lateral_tokens.mean(dim=1)  # (B, d_model)

        # Concatenate
        concat = torch.cat([frontal_pooled, lateral_pooled], dim=1)  # (B, 2*d_model)

        # Residual + LayerNorm
        residual = self.residual_proj(concat)
        fused = self.layer_norm(concat + self.dropout(residual))

        return fused


class MLPHead(nn.Module):
    """
    MLP prediction head: fused vector → hidden layer → 14 measurements.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        out_dim: int = 14,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim)

        Returns:
            (B, out_dim) — predicted measurements (z-scored)
        """
        return self.mlp(x)


class BM2Net(nn.Module):
    """
    BM2-Net: Dual-view body measurement estimation network.

    Input:  3×640×960 composite tensor
            (Ch0 = frontal+lateral masks, Ch1 = height, Ch2 = weight)
    Output: 14 body measurements
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        mlp_hidden: int = 128,
        num_measurements: int = 14,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
    ):
        super().__init__()

        # ── Backbone: MNASNet 1.0 ─────────────────────────────────────
        mnasnet = models.mnasnet1_0(
            weights=models.MNASNet1_0_Weights.DEFAULT if pretrained_backbone else None,
        )
        # Keep only the convolutional feature extractor (remove classifier)
        self.backbone = mnasnet.layers
        self.backbone_out_channels = 1280  # MNASNet1_0 output channels

        # ── Token projection + embeddings ──────────────────────────────
        self.token_projector = TokenProjector(
            in_channels=self.backbone_out_channels,
            d_model=d_model,
            dropout=dropout,
        )

        # ── Cross-view attention ───────────────────────────────────────
        self.cross_view_attn = CrossViewAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # ── Fusion ─────────────────────────────────────────────────────
        self.fusion = FusionBlock(d_model=d_model, dropout=dropout)

        # ── MLP head ───────────────────────────────────────────────────
        self.mlp_head = MLPHead(
            in_dim=self.fusion.fused_dim,
            hidden_dim=mlp_hidden,
            out_dim=num_measurements,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, 640, 960)

        Returns:
            predictions: (B, 14) predicted body measurements (z-scored)
        """
        # Extract backbone features
        features = self.backbone(x)  # (B, 1280, H', W')

        # Split into view tokens + add embeddings
        frontal_tokens, lateral_tokens = self.token_projector(features)

        # Bidirectional cross-view attention
        frontal_attn, lateral_attn = self.cross_view_attn(
            frontal_tokens, lateral_tokens
        )

        # Fuse views
        fused = self.fusion(frontal_attn, lateral_attn)

        # Predict measurements
        predictions = self.mlp_head(fused)

        return predictions


def build_model(config) -> BM2Net:
    """Factory function to create a BM2Net from a TrainConfig."""
    return BM2Net(
        d_model=config.d_model,
        n_heads=config.n_heads,
        mlp_hidden=config.mlp_hidden,
        num_measurements=config.num_measurements,
        dropout=config.dropout,
        pretrained_backbone=True,
    )
