import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

from pytorch_tabnet.tab_network import TabNet


class SharedTabNetEncoder(nn.Module):
    def __init__(self, n_features: int, n_d: int = 64, n_a: int = 64, n_steps: int = 5):
        super().__init__()
        self.n_d = n_d
        self.tabnet = TabNet(
            input_dim=n_features,
            output_dim=n_d,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        # FIX: TabNet.forward() returns (output, M_loss).
        # Previously the whole tuple was returned and passed directly into the
        # task heads, which silently received a tuple instead of a tensor,
        # causing shape errors (or wrong outputs if PyTorch broadcast it).
        output, M_loss = self.tabnet(x)
        return output, M_loss           # unpack here; caller uses both


class TriageHead(nn.Module):
    def __init__(self, n_d: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_d, 128),
            nn.ReLU(),
            nn.Dropout(0.1),            # small dropout for generalisation
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RemediationHead(nn.Module):
    def __init__(self, n_d: int, n_rem: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_d, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_rem),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskTabNet(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_rem: int,
        n_d: int = 64,          # FIX: was hardcoded to 64; now a proper param
        n_a: int = 64,
        n_steps: int = 5,
        lambda_sparse: float = 1e-3,
    ):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.encoder = SharedTabNetEncoder(n_features, n_d=n_d, n_a=n_a, n_steps=n_steps)
        self.triage = TriageHead(n_d, n_classes)
        self.remediation = RemediationHead(n_d, n_rem)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # FIX: correctly unpack encoder output before passing to heads
        enc, M_loss = self.encoder(x)
        triage_logits = self.triage(enc)
        rem_logits = self.remediation(enc)
        return triage_logits, rem_logits, M_loss

    def compute_loss(
        self,
        triage_logits: torch.Tensor,
        rem_logits: torch.Tensor,
        M_loss: torch.Tensor,
        y_triage: torch.Tensor,
        y_rem: torch.Tensor,
        triage_weight: float = 1.0,
        rem_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        Combined multi-task loss:
            total = triage_CE + rem_CE + lambda_sparse * M_loss

        Call this from your training loop instead of computing losses ad-hoc,
        so the sparse-regularisation term (crucial for TabNet attention quality)
        is never forgotten.
        """
        ce = nn.CrossEntropyLoss()
        loss_triage = ce(triage_logits, y_triage)
        loss_rem = ce(rem_logits, y_rem)
        total = (
            triage_weight * loss_triage
            + rem_weight * loss_rem
            + self.lambda_sparse * M_loss
        )
        return total

    def get_feature_importances(self) -> np.ndarray:
        """Return aggregated feature importances from the TabNet attention masks."""
        return self.encoder.tabnet.feature_importances_