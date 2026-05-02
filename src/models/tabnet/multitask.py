from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pytorch_tabnet.tab_network import TabNet
except Exception as e:
    print("[ERROR] Failed to import TabNet from pytorch_tabnet.tab_network")
    print(f"  Root cause: {type(e).__name__}: {e}")
    print("  Verify that both pytorch-tabnet and torch import cleanly.")
    raise


class SharedTabNetEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
    ):
        super().__init__()
        self.tabnet = TabNet(
            input_dim=n_features,
            output_dim=n_d,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            epsilon=1e-15,
            virtual_batch_size=128,
            momentum=0.02,
            group_attention_matrix=torch.eye(n_features),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded, sparsity_loss = self.tabnet(x.float())
        return encoded, sparsity_loss


class TriageHead(nn.Module):
    def __init__(self, n_d: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_d, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RemediationHead(nn.Module):
    def __init__(self, n_d: int, n_remediations: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_d, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_remediations),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskTabNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_triage_classes: int,
        n_remediations: int,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        mode: str = "multitask",
    ):
        super().__init__()
        self.mode = mode
        self.encoder = SharedTabNetEncoder(
            n_features=n_features,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
        )
        self.triage_head = TriageHead(n_d, n_triage_classes)
        self.remediation_head = RemediationHead(n_d, n_remediations)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]:
        encoded, sparsity_loss = self.encoder(x)

        triage_logits: Optional[torch.Tensor] = None
        remediation_logits: Optional[torch.Tensor] = None

        if self.mode != "remediation_only":
            triage_logits = self.triage_head(encoded)
        if self.mode != "triage_only":
            remediation_logits = self.remediation_head(encoded)

        return triage_logits, remediation_logits, sparsity_loss

    def predict_proba(
        self,
        x: torch.Tensor,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        self.eval()
        with torch.no_grad():
            triage_logits, remediation_logits, _ = self.forward(x)
            triage_proba = (
                torch.softmax(triage_logits, dim=1).cpu().numpy()
                if triage_logits is not None else None
            )
            remediation_proba = (
                torch.sigmoid(remediation_logits).cpu().numpy()
                if remediation_logits is not None else None
            )
        return triage_proba, remediation_proba

    def predict_triage(self, x: torch.Tensor) -> np.ndarray:
        triage_proba, _ = self.predict_proba(x)
        if triage_proba is None:
            raise RuntimeError("Triage predictions are unavailable in remediation_only mode")
        return np.argmax(triage_proba, axis=1)

    def predict_remediations(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> np.ndarray:
        _, remediation_proba = self.predict_proba(x)
        if remediation_proba is None:
            raise RuntimeError("Remediation predictions are unavailable in triage_only mode")
        return (remediation_proba >= threshold).astype(int)

    def rank_remediations(
        self,
        x: torch.Tensor,
        top_k: Optional[int] = None,
    ) -> List[np.ndarray]:
        _, remediation_proba = self.predict_proba(x)
        if remediation_proba is None:
            raise RuntimeError("Remediation rankings are unavailable in triage_only mode")

        ranked: List[np.ndarray] = []
        for probs in remediation_proba:
            indices = np.argsort(-probs)
            if top_k is not None:
                indices = indices[:top_k]
            ranked.append(indices)
        return ranked


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        triage_weight: float = 1.0,
        remediation_weight: float = 1.0,
        class_weights: Optional[np.ndarray] = None,
        remediation_pos_weight: Optional[np.ndarray] = None,
        remediation_loss_type: str = "bce",
        remediation_focal_gamma: float = 2.0,
        mode: str = "multitask",
    ):
        super().__init__()
        self.triage_weight = triage_weight
        self.remediation_weight = remediation_weight
        self.remediation_loss_type = remediation_loss_type
        self.remediation_focal_gamma = remediation_focal_gamma
        self.mode = mode

        if class_weights is not None:
            self.triage_loss_fn = nn.CrossEntropyLoss(
                weight=torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.triage_loss_fn = nn.CrossEntropyLoss()

        if remediation_pos_weight is not None:
            pos_weight = torch.tensor(remediation_pos_weight, dtype=torch.float32)
            self.remediation_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.register_buffer("remediation_pos_weight", pos_weight)
        else:
            self.remediation_loss_fn = nn.BCEWithLogitsLoss()
            self.remediation_pos_weight = None

    def forward(
        self,
        triage_logits: Optional[torch.Tensor],
        remediation_logits: Optional[torch.Tensor],
        y_triage: torch.Tensor,
        y_remediation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = y_triage.device
        triage_loss = torch.tensor(0.0, device=device)
        remediation_loss = torch.tensor(0.0, device=device)

        if self.mode != "remediation_only" and triage_logits is not None:
            triage_loss = self.triage_loss_fn(triage_logits, y_triage)

        if self.mode != "triage_only" and remediation_logits is not None:
            if self.remediation_loss_type == "bce":
                remediation_loss = self.remediation_loss_fn(
                    remediation_logits,
                    y_remediation.float(),
                )
            elif self.remediation_loss_type == "focal":
                bce_per_label = nn.functional.binary_cross_entropy_with_logits(
                    remediation_logits,
                    y_remediation.float(),
                    pos_weight=self.remediation_pos_weight,
                    reduction="none",
                )
                probabilities = torch.sigmoid(remediation_logits)
                p_t = (
                    probabilities * y_remediation.float()
                    + (1.0 - probabilities) * (1.0 - y_remediation.float())
                )
                focal_factor = (1.0 - p_t).pow(self.remediation_focal_gamma)
                remediation_loss = (focal_factor * bce_per_label).mean()
            else:
                raise ValueError("remediation_loss_type must be 'bce' or 'focal'")

        if self.mode == "triage_only":
            total_loss = self.triage_weight * triage_loss
        elif self.mode == "remediation_only":
            total_loss = self.remediation_weight * remediation_loss
        else:
            total_loss = (
                self.triage_weight * triage_loss
                + self.remediation_weight * remediation_loss
            )

        return total_loss, triage_loss, remediation_loss


def create_multitask_model(
    n_features: int,
    n_triage_classes: int,
    n_remediations: int,
    device: str = "cpu",
    mode: str = "multitask",
    verbose: bool = True,
) -> MultiTaskTabNet:
    """Create the shared-encoder TabNet multitask model."""

    model = MultiTaskTabNet(
        n_features=n_features,
        n_triage_classes=n_triage_classes,
        n_remediations=n_remediations,
        mode=mode,
    ).to(device)

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("=" * 60)
        print("Creating Multi-Task TabNet Model")
        print("=" * 60)
        print(f"  [OK] Mode: {mode}")
        print(f"  [OK] Device: {device}")
        print(f"  [OK] Total parameters: {total_params:,}")
        print(f"  [OK] Trainable parameters: {trainable_params:,}")

    return model
