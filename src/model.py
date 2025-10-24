"""Neural network models for copy-move forgery detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict


class DinoBackbone(nn.Module):
    """
    DINOv2 Vision Transformer backbone for feature extraction.

    Supports frozen or fine-tunable modes.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        freeze: bool = True,
        pretrained: bool = True
    ):
        """
        Args:
            model_name: DINOv2 model variant (vits14, vitb14, vitl14, vitg14)
            freeze: Whether to freeze backbone weights
            pretrained: Load pretrained weights
        """
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze

        # Load DINOv2 model
        # Note: In production, load from local weights file
        # For now, we'll use a placeholder that can be replaced
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        except Exception:
            # Fallback: create a simple conv backbone for testing
            print(f"Warning: Could not load {model_name}, using simple conv backbone")
            self.backbone = self._create_simple_backbone()

        # Get feature dimension
        if hasattr(self.backbone, 'embed_dim'):
            self.feat_dim = self.backbone.embed_dim
        else:
            self.feat_dim = 384  # Default for ViT-S

        # Freeze if requested
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def _create_simple_backbone(self):
        """Create simple ConvNet backbone for testing when DINOv2 unavailable."""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input image.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            Features (B, C, H', W') where H', W' are downsampled
        """
        if self.freeze:
            self.backbone.eval()
            with torch.no_grad():
                feats = self._extract_features(x)
        else:
            feats = self._extract_features(x)

        return feats

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial features from backbone."""
        # Check if using ViT or Conv backbone
        if hasattr(self.backbone, 'get_intermediate_layers'):
            # DINOv2 ViT - extract patch features
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            # out: (B, N_patches + 1, C) where +1 is CLS token

            # Remove CLS token (first token)
            B, N_total, C = out.shape
            patch_tokens = out[:, 1:, :]  # (B, N_patches, C)

            # Calculate spatial dimensions from input size and patch size
            # DINOv2 vits14 has patch_size=14
            patch_size = 14
            _, _, H_in, W_in = x.shape
            H = H_in // patch_size
            W = W_in // patch_size

            # Reshape to spatial grid
            feats = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        else:
            # Simple conv backbone
            feats = self.backbone(x)

        return feats


class CorrHead(nn.Module):
    """
    Correlation head that processes self-correlation maps.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128
    ):
        """
        Args:
            in_channels: Number of input correlation channels (typically top_k)
            hidden_channels: Hidden layer size
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, 1, 1)

    def forward(self, corr_map: torch.Tensor) -> torch.Tensor:
        """
        Process correlation map to saliency map.

        Args:
            corr_map: Correlation map (B, K, H, W)

        Returns:
            Saliency map (B, 1, H, W)
        """
        x = F.relu(self.bn1(self.conv1(corr_map)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x


class StripPooling(nn.Module):
    """
    Strip Pooling for global context (horizontal + vertical).

    Inspired by CMFDFormer's PCSD module.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # Pool to (H, 1)
        self.pool_v = nn.AdaptiveAvgPool2d((1, None))  # Pool to (1, W)
        self.conv_h = nn.Conv2d(in_channels, in_channels, (1, 1))
        self.conv_v = nn.Conv2d(in_channels, in_channels, (1, 1))
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, x):
        # Horizontal pooling and broadcast
        h_pool = self.pool_h(x)  # (B, C, H, 1)
        h_pool = self.conv_h(h_pool)
        h_pool = h_pool.expand_as(x)

        # Vertical pooling and broadcast
        v_pool = self.pool_v(x)  # (B, C, 1, W)
        v_pool = self.conv_v(v_pool)
        v_pool = v_pool.expand_as(x)

        # Fuse
        fused = self.fusion(torch.cat([h_pool, v_pool], dim=1))
        return x + fused


class TinyDecoder(nn.Module):
    """
    Lightweight decoder for refining correlation maps into masks.

    Uses a simple U-Net style architecture with optional strip pooling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        backbone_channels: int = 384,
        hidden_channels: int = 64,
        use_strip_pool: bool = False
    ):
        """
        Args:
            in_channels: Input channels from correlation head
            backbone_channels: Channels from backbone features (for skip connections)
            hidden_channels: Hidden layer size
            use_strip_pool: Use strip pooling for global context
        """
        super().__init__()

        self.use_strip_pool = use_strip_pool

        # Encoder
        self.enc1 = self._conv_block(in_channels, hidden_channels)
        self.enc2 = self._conv_block(hidden_channels, hidden_channels * 2)

        # Bottleneck with skip from backbone
        self.bottleneck = self._conv_block(
            hidden_channels * 2 + backbone_channels,
            hidden_channels * 2
        )

        # Decoder
        self.dec1 = self._upconv_block(hidden_channels * 2, hidden_channels)
        self.dec2 = self._upconv_block(hidden_channels, hidden_channels // 2)

        # Strip pooling before output
        if use_strip_pool:
            self.strip_pool = StripPooling(hidden_channels // 2)

        # Output
        self.out_conv = nn.Conv2d(hidden_channels // 2, 1, 1)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Convolutional block with BN and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _upconv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Upsampling block."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(
        self,
        saliency: torch.Tensor,
        backbone_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decode saliency map to binary mask.

        Args:
            saliency: Saliency map from CorrHead (B, 1, H, W)
            backbone_feat: Optional backbone features for skip (B, C, H, W)

        Returns:
            Logits (B, 1, H_out, W_out)
        """
        # Encoder
        e1 = self.enc1(saliency)  # (B, hidden, H, W)
        e2 = self.enc2(F.max_pool2d(e1, 2))  # (B, hidden*2, H/2, W/2)

        # Bottleneck with skip
        if backbone_feat is not None:
            # Resize backbone features to match e2
            bb_resized = F.interpolate(
                backbone_feat, size=e2.shape[2:],
                mode='bilinear', align_corners=False
            )
            bottleneck_in = torch.cat([e2, bb_resized], dim=1)
        else:
            bottleneck_in = e2

        b = self.bottleneck(bottleneck_in)  # (B, hidden*2, H/2, W/2)

        # Decoder
        d1 = self.dec1(b)  # (B, hidden, H, W)
        d2 = self.dec2(d1)  # (B, hidden/2, H*2, W*2)

        # Strip pooling for global context
        if self.use_strip_pool:
            d2 = self.strip_pool(d2)

        # Output
        out = self.out_conv(d2)  # (B, 1, H*2, W*2)

        return out


class CMFDNet(nn.Module):
    """
    Complete Copy-Move Forgery Detection Network.

    Combines backbone, correlation computation, and decoder.
    """

    def __init__(
        self,
        backbone: str = "dinov2_vits14",
        freeze_backbone: bool = True,
        patch: int = 12,
        stride: int = 4,
        top_k: int = 5,
        use_decoder: bool = True,
        use_strip_pool: bool = False
    ):
        """
        Args:
            backbone: Backbone model name
            freeze_backbone: Whether to freeze backbone
            patch: Patch size for correlation
            stride: Stride for correlation
            top_k: Number of top matches
            use_decoder: Whether to use decoder (False = correlation only)
            use_strip_pool: Use strip pooling in decoder for global context
        """
        super().__init__()

        self.patch = patch
        self.stride = stride
        self.top_k = top_k
        self.use_decoder = use_decoder

        # Backbone
        self.backbone = DinoBackbone(
            model_name=backbone,
            freeze=freeze_backbone
        )

        # Correlation head
        self.corr_head = CorrHead(in_channels=top_k)

        # Decoder
        if use_decoder:
            self.decoder = TinyDecoder(
                in_channels=1,
                backbone_channels=self.backbone.feat_dim,
                use_strip_pool=use_strip_pool
            )
        else:
            self.decoder = None

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input image (B, 3, H, W)
            return_features: Whether to return intermediate features

        Returns:
            Dictionary with 'logits' and optionally 'features', 'corr_map'
        """
        # Extract features
        feats = self.backbone(x)  # (B, C, H', W')

        # Compute self-correlation (this should use corr.py)
        # For now, create placeholder correlation map
        # In practice, call self_corr from corr.py here
        from corr import self_corr

        corr_result = self_corr(
            feats,
            patch=self.patch,
            stride=self.stride,
            top_k=self.top_k
        )
        corr_map = corr_result['corr_map']  # (B, k, H, W)

        # Process correlation
        saliency = self.corr_head(corr_map)  # (B, 1, H, W)

        # Decode
        if self.use_decoder:
            logits = self.decoder(saliency, feats)
        else:
            logits = saliency

        # Upsample to input size
        logits = F.interpolate(
            logits, size=x.shape[2:],
            mode='bilinear', align_corners=False
        )

        output = {'logits': logits}

        if return_features:
            output['features'] = feats
            output['corr_map'] = corr_map
            output['saliency'] = saliency

        return output

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict binary mask.

        Args:
            x: Input image (B, 3, H, W)

        Returns:
            Binary mask (B, 1, H, W)
        """
        with torch.no_grad():
            output = self.forward(x)
            logits = output['logits']
            mask = (torch.sigmoid(logits) > 0.5).float()

        return mask
