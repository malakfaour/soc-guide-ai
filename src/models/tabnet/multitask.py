"""
Multi-task TabNet model for triage and remediation.

Combines triage classification and multi-label remediation prediction
using a shared TabNet encoder with separate task-specific heads.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Tuple, Dict, List, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "training"))

try:
    from pytorch_tabnet.tab_network import TabNet
except Exception as e:
    print("[ERROR] Failed to import TabNet from pytorch_tabnet.tab_network")
    print(f"  Root cause: {type(e).__name__}: {e}")
    print("  Verify that both pytorch-tabnet and torch import cleanly.")
    sys.exit(1)


class SharedTabNetEncoder(nn.Module):
    """
    Shared TabNet encoder for multi-task learning.
    
    Outputs intermediate representations that are used by both
    triage classification and remediation prediction heads.
    """
    
    def __init__(
        self,
        n_features: int,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
        lambda_sparse: float = 1e-3,
    ):
        """
        Initialize shared TabNet encoder.
        
        Parameters
        ----------
        n_features : int
            Number of input features
        n_d : int
            Width of decision step features
        n_a : int
            Width of attention features
        n_steps : int
            Number of decision steps
        gamma : float
            Feature reuse coefficient
        n_independent : int
            Independent components
        n_shared : int
            Shared components
        lambda_sparse : float
            Sparsity regularization
        """
        super().__init__()
        
        self.n_features = n_features
        self.n_d = n_d
        
        # TabNet encoder (shared backbone)
        self.tabnet = TabNet(
            input_dim=n_features,
            output_dim=n_d,  # Output dimension for downstream tasks
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            lambda_sparse=lambda_sparse,
            epsilon=1e-15,
            virtual_batch_size=None,
            momentum=0.02,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through shared encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features (batch_size, n_features)
        
        Returns
        -------
        torch.Tensor
            Encoded features (batch_size, n_d)
        """
        encoded, _ = self.tabnet(x)
        return encoded


class TriageHead(nn.Module):
    """
    Triage classification head for multi-class prediction.
    """
    
    def __init__(self, n_d: int, n_classes: int):
        """
        Initialize triage head.
        
        Parameters
        ----------
        n_d : int
            Input dimension from encoder
        n_classes : int
            Number of triage classes
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(n_d, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through triage head.
        
        Parameters
        ----------
        x : torch.Tensor
            Encoded features (batch_size, n_d)
        
        Returns
        -------
        torch.Tensor
            Logits (batch_size, n_classes)
        """
        return self.mlp(x)


class RemediationHead(nn.Module):
    """
    Multi-label remediation prediction head.
    
    Each remediation is treated as independent binary classification.
    """
    
    def __init__(self, n_d: int, n_remediations: int):
        """
        Initialize remediation head.
        
        Parameters
        ----------
        n_d : int
            Input dimension from encoder
        n_remediations : int
            Number of remediation actions
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(n_d, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_remediations),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through remediation head.
        
        Parameters
        ----------
        x : torch.Tensor
            Encoded features (batch_size, n_d)
        
        Returns
        -------
        torch.Tensor
            Logits for each remediation (batch_size, n_remediations)
        """
        return self.mlp(x)


class MultiTaskTabNet(nn.Module):
    """
    Multi-task learning model combining triage and remediation.
    
    Architecture:
    - Shared TabNet encoder
    - Triage head (multi-class classification)
    - Remediation head (multi-label binary classification)
    """
    
    def __init__(
        self,
        n_features: int,
        n_triage_classes: int,
        n_remediations: int,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
    ):
        """
        Initialize multi-task TabNet model.
        
        Parameters
        ----------
        n_features : int
            Number of input features
        n_triage_classes : int
            Number of triage classes
        n_remediations : int
            Number of remediation actions
        n_d : int
            Decision feature width
        n_a : int
            Attention feature width
        n_steps : int
            Number of decision steps
        gamma : float
            Feature reuse coefficient
        """
        super().__init__()
        
        self.n_features = n_features
        self.n_triage_classes = n_triage_classes
        self.n_remediations = n_remediations
        
        # Shared encoder
        self.encoder = SharedTabNetEncoder(
            n_features=n_features,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
        )
        
        # Task-specific heads
        self.triage_head = TriageHead(n_d, n_triage_classes)
        self.remediation_head = RemediationHead(n_d, n_remediations)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multi-task model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features (batch_size, n_features)
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Triage logits (batch_size, n_classes)
            Remediation logits (batch_size, n_remediations)
        """
        # Shared encoding
        encoded = self.encoder(x)
        
        # Task predictions
        triage_logits = self.triage_head(encoded)
        remediation_logits = self.remediation_head(encoded)
        
        return triage_logits, remediation_logits
    
    def predict_proba(
        self,
        x: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get probability predictions.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features (batch_size, n_features)
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Triage probabilities (batch_size, n_classes)
            Remediation probabilities (batch_size, n_remediations)
        """
        self.eval()
        with torch.no_grad():
            triage_logits, remediation_logits = self.forward(x)
            
            # Convert logits to probabilities
            triage_proba = torch.softmax(triage_logits, dim=1).cpu().numpy()
            remediation_proba = torch.sigmoid(remediation_logits).cpu().numpy()
        
        return triage_proba, remediation_proba
    
    def predict_triage(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict triage class.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features (batch_size, n_features)
        
        Returns
        -------
        np.ndarray
            Predicted class indices (batch_size,)
        """
        triage_proba, _ = self.predict_proba(x)
        return np.argmax(triage_proba, axis=1)
    
    def predict_remediations(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predict remediation actions.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features (batch_size, n_features)
        threshold : float
            Probability threshold for remediation selection
        
        Returns
        -------
        np.ndarray
            Binary predictions for each remediation (batch_size, n_remediations)
        """
        _, remediation_proba = self.predict_proba(x)
        return (remediation_proba >= threshold).astype(int)
    
    def rank_remediations(
        self,
        x: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Rank remediation actions by probability.
        
        Parameters
        ----------
        x : torch.Tensor
            Input features (batch_size, n_features)
        top_k : int, optional
            Return only top-k remediations (None = all)
        
        Returns
        -------
        List[np.ndarray]
            For each sample, indices sorted by probability (descending)
        """
        _, remediation_proba = self.predict_proba(x)
        ranked = []
        
        for probs in remediation_proba:
            # Sort indices by probability (descending)
            indices = np.argsort(-probs)
            if top_k:
                indices = indices[:top_k]
            ranked.append(indices)
        
        return ranked


class MultiTaskLoss(nn.Module):
    """
    Combined loss function for multi-task learning.
    
    Combines:
    - Cross-entropy for triage classification
    - Binary cross-entropy for multi-label remediations
    """
    
    def __init__(
        self,
        triage_weight: float = 1.0,
        remediation_weight: float = 1.0,
        class_weights: Optional[np.ndarray] = None,
    ):
        """
        Initialize multi-task loss.
        
        Parameters
        ----------
        triage_weight : float
            Weight for triage classification loss
        remediation_weight : float
            Weight for remediation prediction loss
        class_weights : np.ndarray, optional
            Class weights for triage loss (n_classes,)
        """
        super().__init__()
        
        self.triage_weight = triage_weight
        self.remediation_weight = remediation_weight
        
        # Triage loss (multi-class)
        if class_weights is not None:
            class_weights_t = torch.tensor(class_weights, dtype=torch.float32)
            self.triage_loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
        else:
            self.triage_loss_fn = nn.CrossEntropyLoss()
        
        # Remediation loss (multi-label)
        self.remediation_loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        triage_logits: torch.Tensor,
        remediation_logits: torch.Tensor,
        y_triage: torch.Tensor,
        y_remediation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss.
        
        Parameters
        ----------
        triage_logits : torch.Tensor
            Triage predictions (batch_size, n_classes)
        remediation_logits : torch.Tensor
            Remediation predictions (batch_size, n_remediations)
        y_triage : torch.Tensor
            Triage targets (batch_size,)
        y_remediation : torch.Tensor
            Remediation targets (batch_size, n_remediations)
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Total loss, triage loss, remediation loss
        """
        # Triage loss
        triage_loss = self.triage_loss_fn(triage_logits, y_triage)
        
        # Remediation loss
        remediation_loss = self.remediation_loss_fn(
            remediation_logits,
            y_remediation.float()
        )
        
        # Combined loss
        total_loss = (
            self.triage_weight * triage_loss +
            self.remediation_weight * remediation_loss
        )
        
        return total_loss, triage_loss, remediation_loss


def create_multitask_model(
    n_features: int,
    n_triage_classes: int,
    n_remediations: int,
    device: str = "cpu",
    verbose: bool = True,
) -> MultiTaskTabNet:
    """
    Create and initialize multi-task TabNet model.
    
    Parameters
    ----------
    n_features : int
        Number of input features
    n_triage_classes : int
        Number of triage classes
    n_remediations : int
        Number of remediation actions
    device : str
        Device to run model on ('cpu' or 'cuda')
    verbose : bool
        Print model information
    
    Returns
    -------
    MultiTaskTabNet
        Initialized model
    """
    if verbose:
        print("=" * 60)
        print("Creating Multi-Task TabNet Model")
        print("=" * 60)
        print(f"\n[CONFIG] Model architecture:")
        print(f"  Input features: {n_features}")
        print(f"  Triage classes: {n_triage_classes}")
        print(f"  Remediation actions: {n_remediations}")
        print(f"  Device: {device}")
    
    model = MultiTaskTabNet(
        n_features=n_features,
        n_triage_classes=n_triage_classes,
        n_remediations=n_remediations,
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
    )
    
    model = model.to(device)
    
    if verbose:
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        print(f"\n[PARAMETERS]")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print("\n" + "=" * 60)
        print("✓ Model created successfully!")
        print("=" * 60)
    
    return model


if __name__ == "__main__":
    """Test multi-task model"""
    
    print("\nTesting Multi-Task TabNet Model\n")
    
    # Model configuration
    n_features = 44
    n_triage_classes = 3
    n_remediations = 5  # Example: 5 remediation actions
    
    try:
        # Create model
        model = create_multitask_model(
            n_features=n_features,
            n_triage_classes=n_triage_classes,
            n_remediations=n_remediations,
            device="cpu",
            verbose=True
        )
        
        # Create test batch
        print("\n[TEST] Forward pass with test data...")
        batch_size = 32
        x = torch.randn(batch_size, n_features)
        
        # Forward pass
        triage_logits, remediation_logits = model(x)
        
        print(f"  ✓ Input shape: {x.shape}")
        print(f"  ✓ Triage logits shape: {triage_logits.shape}")
        print(f"  ✓ Remediation logits shape: {remediation_logits.shape}")
        
        # Test predictions
        print("\n[TEST] Prediction methods...")
        triage_proba, remediation_proba = model.predict_proba(x)
        print(f"  ✓ Triage probabilities: {triage_proba.shape}")
        print(f"  ✓ Remediation probabilities: {remediation_proba.shape}")
        
        triage_pred = model.predict_triage(x)
        print(f"  ✓ Triage predictions: {triage_pred.shape}")
        
        remediation_pred = model.predict_remediations(x)
        print(f"  ✓ Remediation binary predictions: {remediation_pred.shape}")
        
        ranked = model.rank_remediations(x, top_k=3)
        print(f"  ✓ Ranked remediations: {len(ranked)} samples")
        print(f"    Example (top-3): {ranked[0]}")
        
        # Test loss function
        print("\n[TEST] Loss function...")
        criterion = MultiTaskLoss(
            triage_weight=1.0,
            remediation_weight=1.0,
        )
        
        y_triage = torch.randint(0, n_triage_classes, (batch_size,))
        y_remediation = torch.randint(0, 2, (batch_size, n_remediations))
        
        total_loss, triage_loss, remediation_loss = criterion(
            triage_logits, remediation_logits, y_triage, y_remediation
        )
        
        print(f"  ✓ Total loss: {total_loss.item():.4f}")
        print(f"  ✓ Triage loss: {triage_loss.item():.4f}")
        print(f"  ✓ Remediation loss: {remediation_loss.item():.4f}")
        
        # Test backward pass
        print("\n[TEST] Backward pass...")
        optimizer = Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print("  ✓ Gradient computation successful")
        
        print("\n" + "=" * 60)
        print("✓ All multi-task model tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
