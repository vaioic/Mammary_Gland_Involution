"""
Attention-based Multiple Instance Learning (MIL) Model
for Whole Slide Image Classification with Interpretability

This architecture:
1. Extracts features from patches using a backbone (ResNet/ViT)
2. Uses attention mechanism to aggregate patch features
3. Attention weights provide interpretability (heatmaps)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional


class FeatureExtractor(nn.Module):
    """
    Extract features from image patches using pre-trained backbone.
    """
    
    def __init__(self, backbone, pretrained, freeze_backbone):
        super().__init__()
        self.backbone_name = backbone
        
        if backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            # Remove final classification layer
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 2048
            
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 512
            
        elif backbone == 'vit_b_16':
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            base_model.heads = nn.Identity()
            self.features = base_model
            self.feature_dim = 768
            
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        if self.backbone_name == 'vit_b_16':
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return features


class AttentionMIL(nn.Module):
    """
    Attention-based MIL for slide-level classification.
    
    The attention weights indicate which patches are most important,
    providing interpretability for the classification decision.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 8,
        dropout: float = 0.25,
        attention_branches: int = 1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.attention_branches = attention_branches
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, attention_branches)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * attention_branches, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: (batch_size, num_patches, feature_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: (batch_size, num_classes)
            attention_weights: (batch_size, num_patches) if return_attention=True
        """
        # Calculate attention scores
        attention_scores = self.attention(features)  # (batch_size, num_patches, attention_branches)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, num_patches, attention_branches)
        
        # Weighted aggregation of features
        # For each attention branch, compute weighted sum
        aggregated_features = []
        for i in range(self.attention_branches):
            weights = attention_weights[:, :, i:i+1]  # (batch_size, num_patches, 1)
            weighted_features = (features * weights).sum(dim=1)  # (batch_size, feature_dim)
            aggregated_features.append(weighted_features)
        
        # Concatenate features from all attention branches
        aggregated_features = torch.cat(aggregated_features, dim=1)  # (batch_size, feature_dim * branches)
        
        # Classification
        logits = self.classifier(aggregated_features)
        
        if return_attention:
            # Return mean attention across branches for visualization
            attention_viz = attention_weights.mean(dim=2)  # (batch_size, num_patches)
            return logits, attention_viz
        
        return logits, None


class GatedAttentionMIL(nn.Module):
    """
    Gated Attention MIL - more sophisticated attention mechanism.
    Implements the architecture from "Attention-based Deep Multiple Instance Learning"
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 8,
        dropout: float = 0.25
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Attention mechanism with gating
        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.attention_weights = nn.Linear(hidden_dim, 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: (batch_size, num_patches, feature_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: (batch_size, num_classes)
            attention_weights: (batch_size, num_patches) if return_attention=True
        """
        # Gated attention
        attention_v = self.attention_V(features)  # (batch_size, num_patches, hidden_dim)
        attention_u = self.attention_U(features)  # (batch_size, num_patches, hidden_dim)
        
        # Element-wise multiplication (gating)
        attention = attention_v * attention_u  # (batch_size, num_patches, hidden_dim)
        
        # Calculate attention scores
        attention_scores = self.attention_weights(attention)  # (batch_size, num_patches, 1)
        attention_scores = attention_scores.squeeze(-1)  # (batch_size, num_patches)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, num_patches)
        
        # Weighted aggregation
        weighted_features = (features * attention_weights.unsqueeze(-1)).sum(dim=1)  # (batch_size, feature_dim)
        
        # Classification
        logits = self.classifier(weighted_features)
        
        if return_attention:
            return logits, attention_weights
        
        return logits, None


class WSIClassifier(nn.Module):
    """
    Complete WSI classifier: Feature Extractor + Attention MIL
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        mil_type: str = 'gated',  # 'simple' or 'gated'
        hidden_dim: int = 256,
        num_classes: int = 8,
        dropout: float = 0.25
    ):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone
        )
        
        # MIL aggregator
        if mil_type == 'simple':
            self.mil = AttentionMIL(
                feature_dim=self.feature_extractor.feature_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout
            )
        elif mil_type == 'gated':
            self.mil = GatedAttentionMIL(
                feature_dim=self.feature_extractor.feature_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown MIL type: {mil_type}")
    
    def forward(
        self,
        patches: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            patches: (batch_size, num_patches, channels, height, width)
            return_attention: Whether to return attention weights
            
        Returns:
            logits: (batch_size, num_classes)
            attention_weights: (batch_size, num_patches) if return_attention=True
        """
        batch_size, num_patches = patches.shape[:2]
        
        # Reshape patches for batch processing
        patches = patches.view(-1, *patches.shape[2:])  # (batch_size * num_patches, C, H, W)
        
        # Extract features
        features = self.feature_extractor(patches)  # (batch_size * num_patches, feature_dim)
        
        # Reshape back
        features = features.view(batch_size, num_patches, -1)  # (batch_size, num_patches, feature_dim)
        
        # MIL aggregation and classification
        logits, attention_weights = self.mil(features, return_attention=return_attention)
        
        return logits, attention_weights


def create_model(
    backbone: str = 'resnet50',
    num_classes: int = 8,
    pretrained: bool = True,
    mil_type: str = 'gated'
) -> WSIClassifier:
    """
    Factory function to create WSI classifier.
    
    Args:
        backbone: Feature extractor backbone ('resnet50', 'resnet34', 'vit_b_16')
        num_classes: Number of classification classes
        pretrained: Whether to use pretrained weights
        mil_type: Type of MIL ('simple' or 'gated')
    
    Returns:
        WSIClassifier model
    """
    model = WSIClassifier(
        backbone=backbone,
        pretrained=pretrained,
        freeze_backbone=False,
        mil_type=mil_type,
        hidden_dim=256,
        num_classes=num_classes,
        dropout=0.25
    )
    
    return model


if __name__ == "__main__":
    # Test model
    model = create_model(backbone='resnet50', num_classes=8, mil_type='gated')
    
    # Dummy input: batch_size=2, num_patches=100, 3x256x256 patches
    dummy_patches = torch.randn(2, 100, 3, 256, 256)
    
    # Forward pass
    logits, attention = model(dummy_patches, return_attention=True)
    
    print(f"Input shape: {dummy_patches.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Attention weights shape: {attention.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
