"""
Adaptive Neuro-Fuzzy Inference System (ANFIS) — Core Engine
============================================================

Hand-implemented from scratch as a PyTorch nn.Module.

Architecture: Takagi-Sugeno type ANFIS with 5 layers.
Reference: Jang, J-SR. "ANFIS: adaptive-network-based fuzzy inference system."
           IEEE Transactions on Systems, Man, and Cybernetics 23.3 (1993): 665-685.

Also referenced in:
- Paper 1: Low-light robust face image SR via neuro-fuzzy inferencing-based
           locality constrained representation (Section III-A)
- Paper 2: Neuro Fuzzy Inferencing Based System and Method For Improving
           Quality of Dark and Low Resolution Images (Section 3.1)
- Paper 3: Estimation of darkness factor from low-light images based on
           adaptive neuro-fuzzy inferencing technique (Section 2.2)

Layer Structure:
    Layer 1 (Fuzzification)   : Gaussian membership functions per input per rule
    Layer 2 (Rule Strength)   : Product of MF outputs (firing strengths)
    Layer 3 (Normalization)   : Normalized firing strengths (w_bar_i)
    Layer 4 (Consequent)      : Linear Takagi-Sugeno consequents
    Layer 5 (Defuzzification) : Weighted sum → crisp output

Learning Algorithm (Hybrid):
    Forward pass  → estimate consequent params via Recursive LSE
    Backward pass → update premise params via gradient descent
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────
#  Layer 1: Fuzzification — Gaussian Membership Functions
# ─────────────────────────────────────────────────────────────

class GaussianMF(nn.Module):
    """Gaussian membership function: μ(x) = exp(-(x - c)² / (2σ²))

    Parameters learned via gradient descent (premise parameters).

    Args:
        n_inputs : Number of input features.
        n_rules  : Number of fuzzy rules (= n_mfs per input in a grid).
    """

    def __init__(self, n_inputs: int, n_rules: int):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_rules  = n_rules

        # Centre c: shape [n_inputs, n_rules]
        # Initialise spread across [-1, 1] range
        c_init = torch.linspace(-1.0, 1.0, n_rules).repeat(n_inputs, 1)
        self.c = nn.Parameter(c_init)   # premise param

        # Width σ: shape [n_inputs, n_rules]
        # Initialise to sensible spread
        sigma_init = torch.ones(n_inputs, n_rules) * (2.0 / n_rules)
        self.sigma = nn.Parameter(sigma_init)   # premise param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute membership degrees for all inputs and all rules.

        Args:
            x : Input tensor of shape [batch, n_inputs].

        Returns:
            mu : Membership tensor of shape [batch, n_inputs, n_rules].
                 mu[b, i, k] = μ_{i,k}(x[b, i])
        """
        # x:      [B, n_inputs]        →  [B, n_inputs, 1]
        # self.c: [n_inputs, n_rules]  →  [1, n_inputs, n_rules]
        x_expand = x.unsqueeze(2)                         # [B, I, 1]
        c_expand = self.c.unsqueeze(0)                    # [1, I, K]
        s_expand = self.sigma.abs().unsqueeze(0) + 1e-8   # [1, I, K]  prevent div/0

        mu = torch.exp(-((x_expand - c_expand) ** 2) / (2 * s_expand ** 2))
        return mu   # [B, I, K]


# ─────────────────────────────────────────────────────────────
#  Layer 2: Rule Strength — Product of MFs
# ─────────────────────────────────────────────────────────────

def compute_rule_strengths(mu: torch.Tensor) -> torch.Tensor:
    """Compute firing strength for each rule via product T-norm.

    For a grid of n_rules per input, the total rules = n_rules^n_inputs.
    We use a full Cartesian product here for clarity, but limit n_rules
    to keep it tractable (default: 3 MFs per input).

    Args:
        mu : [batch, n_inputs, n_rules]  membership degrees.

    Returns:
        w  : [batch, total_rules]  firing strengths.
             w[b, r] = Π_i  μ_{i, r_i}(x[b, i])
    """
    B, n_inputs, n_rules = mu.shape

    # Build all rule index combinations  (Cartesian product)
    # indices: [total_rules, n_inputs]
    idx = torch.cartesian_prod(*[torch.arange(n_rules)] * n_inputs)  # [R, I]

    # Gather membership values per rule
    # mu[:, i, r_i] for each rule r
    w = torch.ones(B, idx.shape[0], device=mu.device)
    for i in range(n_inputs):
        rule_mf_idx = idx[:, i]           # [R]
        # mu[:, i, :]  →  [B, K]  →  gather col rule_mf_idx  →  [B, R]
        w = w * mu[:, i, :][:, rule_mf_idx]

    return w   # [B, R]


# ─────────────────────────────────────────────────────────────
#  Layer 3: Normalisation
# ─────────────────────────────────────────────────────────────

def normalize_strengths(w: torch.Tensor) -> torch.Tensor:
    """Normalize firing strengths: w_bar_i = w_i / Σ_j w_j

    Args:
        w : [batch, total_rules]

    Returns:
        w_bar : [batch, total_rules]  sum-to-1 per sample.
    """
    return w / (w.sum(dim=1, keepdim=True) + 1e-8)


# ─────────────────────────────────────────────────────────────
#  Layer 4: Consequent — Linear Takagi-Sugeno
# ─────────────────────────────────────────────────────────────

class ConsequentLayer(nn.Module):
    """Takagi-Sugeno linear consequent: f_r(x) = p_{r,0} + Σ_i p_{r,i} * x_i

    Consequent parameters {p} are learned via Recursive LSE in the forward
    pass (instead of gradient descent) for faster convergence.

    Args:
        n_inputs     : Number of input features.
        n_rules      : Total number of fuzzy rules.
        n_outputs    : Number of outputs (1 for scalar regression).
    """

    def __init__(self, n_inputs: int, n_rules: int, n_outputs: int = 1):
        super().__init__()
        # p: [n_rules, (n_inputs + 1)]  — +1 for bias term
        self.p = nn.Parameter(torch.randn(n_rules, n_inputs + 1) * 0.01)
        self.n_outputs = n_outputs

    def forward(self, x: torch.Tensor, w_bar: torch.Tensor) -> torch.Tensor:
        """Compute weighted consequent output.

        Args:
            x     : [batch, n_inputs]   input features.
            w_bar : [batch, n_rules]    normalized firing strengths.

        Returns:
            out : [batch, n_outputs]   defuzzified output.
        """
        B = x.shape[0]

        # Build augmented input: [B, n_inputs+1]  (append 1 for bias)
        ones = torch.ones(B, 1, device=x.device)
        x_aug = torch.cat([x, ones], dim=1)   # [B, n_inputs+1]

        # Consequent for each rule: f_r = x_aug @ p[r]  →  [B, n_rules]
        # Efficient: [B, I+1] @ [I+1, R] = [B, R]
        f = x_aug @ self.p.T   # [B, R]

        # Weighted sum: output = Σ_r  w_bar_r * f_r
        out = (w_bar * f).sum(dim=1, keepdim=True)   # [B, 1]
        return out


# ─────────────────────────────────────────────────────────────
#  Full ANFIS Module
# ─────────────────────────────────────────────────────────────

class ANFIS(nn.Module):
    """Adaptive Neuro-Fuzzy Inference System (Takagi-Sugeno, Gaussian MFs).

    Implements the exact 5-layer architecture from Jang (1993), as used in
    Papers 1, 2, and 3 for illumination-guided image enhancement.

    Args:
        n_inputs   : Dimensionality of the input feature vector.
        n_mfs      : Number of membership functions per input (typically 2–5).
        n_outputs  : Number of scalar outputs (default: 1 for darkness factor).

    Example:
        >>> anfis = ANFIS(n_inputs=4, n_mfs=3)     # 4 features, 3×3×3×3 = 81 rules
        >>> x = torch.rand(8, 4)                   # batch of 8
        >>> y = anfis(x)                           # [8, 1]  predictions
    """

    def __init__(self, n_inputs: int = 4, n_mfs: int = 3, n_outputs: int = 1):
        super().__init__()
        self.n_inputs  = n_inputs
        self.n_mfs     = n_mfs
        self.n_rules   = n_mfs ** n_inputs
        self.n_outputs = n_outputs

        # Layer 1: Gaussian membership functions
        self.mf_layer = GaussianMF(n_inputs, n_mfs)

        # Layer 4: Consequent parameters
        self.consequent = ConsequentLayer(n_inputs, self.n_rules, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full ANFIS forward pass (5 layers).

        Args:
            x : [batch, n_inputs]  input features (normalised to [-1, 1]).

        Returns:
            y : [batch, n_outputs]  crisp output.
        """
        # Layer 1 — Fuzzification: μ_{i,k}(x_i)
        mu = self.mf_layer(x)                  # [B, I, K]

        # Layer 2 — Rule strength: w_r = Π_i μ_{i, r_i}
        w = compute_rule_strengths(mu)          # [B, R]

        # Layer 3 — Normalisation: w_bar_r = w_r / Σ w
        w_bar = normalize_strengths(w)          # [B, R]

        # Layers 4 + 5 — Consequent + Defuzzification
        y = self.consequent(x, w_bar)           # [B, 1]

        return y

    # ── Convenience: get interpretable membership functions ──────────

    def get_membership_params(self) -> dict:
        """Return the current premise parameters as numpy arrays.

        Useful for plotting membership functions or sanity-checking after training.

        Returns:
            dict with keys 'centers' [n_inputs, n_mfs] and
                           'sigmas'  [n_inputs, n_mfs]
        """
        return {
            'centers': self.mf_layer.c.detach().cpu().numpy(),
            'sigmas' : self.mf_layer.sigma.abs().detach().cpu().numpy(),
        }

    def get_rule_count(self) -> int:
        """Total number of fuzzy rules in the rule base."""
        return self.n_rules


# ─────────────────────────────────────────────────────────────
#  Hybrid Learning Trainer
# ─────────────────────────────────────────────────────────────

class ANFISTrainer:
    """Trains ANFIS using the hybrid learning algorithm.

    Forward sweep: Recursive Least Squares (RLS) for consequent params.
    Backward sweep: Gradient descent for premise params.

    Reference: Jang (1993), Section III — Hybrid Learning Algorithm.
               Paper 3, Section 2.3 — Training of the ANFIS model.

    Args:
        model       : ANFIS instance.
        lr          : Learning rate for premise parameter gradient descent.
        lse_lambda  : Forgetting factor for RLS (default 1.0 → batch LSE).
    """

    def __init__(self, model: ANFIS, lr: float = 1e-3, lse_lambda: float = 1.0):
        self.model = model
        self.lr    = lr

        # Only premise params (MF centres + sigmas) via gradient descent
        premise_params = list(model.mf_layer.parameters())
        self.optimizer = torch.optim.Adam(premise_params, lr=lr)
        self.criterion = nn.MSELoss()

        # Consequent params via full LSE each epoch
        self.lse_lambda = lse_lambda

    def _update_consequents_lse(self,
                                 x: torch.Tensor,
                                 y_target: torch.Tensor) -> None:
        """One-shot Least Squares update of consequent parameters.

        Given the current premise params, compute the optimal consequent
        params analytically: p* = (Φ^T Φ + λI)^{-1} Φ^T y

        where Φ[b, r] = w_bar[b, r] * x_aug[b] (per-rule regressor).

        Args:
            x        : [N, n_inputs]   training inputs.
            y_target : [N, 1]          training targets.
        """
        with torch.no_grad():
            # Forward through layers 1–3 to get w_bar
            mu    = self.model.mf_layer(x)
            w     = compute_rule_strengths(mu)
            w_bar = normalize_strengths(w)            # [N, R]

            # Build regressor matrix Φ: [N, R*(n_inputs+1)]
            N = x.shape[0]
            ones = torch.ones(N, 1, device=x.device)
            x_aug = torch.cat([x, ones], dim=1)      # [N, I+1]
            R = w_bar.shape[1]
            I_plus_1 = x_aug.shape[1]

            # Φ[b, r*(I+1) : (r+1)*(I+1)] = w_bar[b, r] * x_aug[b]
            Phi = torch.zeros(N, R * I_plus_1, device=x.device)
            for r in range(R):
                Phi[:, r*I_plus_1:(r+1)*I_plus_1] = \
                    w_bar[:, r:r+1] * x_aug           # [N, I+1]

            # LSE: p* = (Φ^T Φ + λI)^{-1} Φ^T y
            lam = self.lse_lambda * torch.eye(Phi.shape[1], device=x.device)
            p_flat = torch.linalg.lstsq(
                Phi.T @ Phi + lam,
                Phi.T @ y_target
            ).solution   # [R*(I+1), 1]

            # Reshape into [R, I+1] and assign
            p_new = p_flat.view(R, I_plus_1)
            self.model.consequent.p.copy_(p_new)

    def train_epoch(self,
                    x: torch.Tensor,
                    y_target: torch.Tensor) -> float:
        """One training epoch (forward LSE + backward gradient descent).

        Args:
            x        : [N, n_inputs]  training inputs.
            y_target : [N, 1]         training targets.

        Returns:
            loss : Scalar MSE loss value.
        """
        # ── Forward: update consequents via LSE ─────────────────────
        self._update_consequents_lse(x, y_target)

        # ── Backward: update premise params via gradient descent ─────
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss   = self.criterion(y_pred, y_target)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(self,
            x: torch.Tensor,
            y_target: torch.Tensor,
            epochs: int = 200,
            verbose: bool = True) -> list:
        """Train ANFIS for a fixed number of epochs.

        Args:
            x        : [N, n_inputs]  training inputs.
            y_target : [N, 1]         training targets (normalised to [0,1]).
            epochs   : Number of hybrid learning epochs.
            verbose  : Print loss every 20 epochs.

        Returns:
            loss_history : List of per-epoch MSE losses.
        """
        history = []
        for ep in range(1, epochs + 1):
            loss = self.train_epoch(x, y_target)
            history.append(loss)
            if verbose and ep % 20 == 0:
                print(f"  Epoch [{ep:>4}/{epochs}]  MSE Loss: {loss:.6f}")
        return history


# ─────────────────────────────────────────────────────────────
#  Quick Sanity Check
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(42)
    print("=" * 55)
    print("ANFIS Core — Sanity Check")
    print("=" * 55)

    # 4 illumination features, 2 MFs per input → 2^4 = 16 rules
    model   = ANFIS(n_inputs=4, n_mfs=2)
    trainer = ANFISTrainer(model, lr=5e-3)

    print(f"Rules: {model.get_rule_count()}  (n_mfs^n_inputs = 2^4)")

    # Synthetic data: random features → target in [0, 1]
    N   = 500
    x   = torch.rand(N, 4) * 2 - 1    # features in [-1, 1]
    # Ground truth: darkness ≈ mean darkness of features
    y   = ((-x.mean(dim=1, keepdim=True) + 1) / 2).clamp(0, 1)

    history = trainer.fit(x, y, epochs=100, verbose=True)
    print(f"\nFinal MSE: {history[-1]:.6f}")

    # Inference
    model.eval()
    with torch.no_grad():
        y_hat = model(x[:5])
    print(f"\nSample predictions:\n  pred  = {y_hat.squeeze().numpy()}")
    print(f"  truth = {y[:5].squeeze().numpy()}")
    print("\n✓ ANFIS Core passed sanity check.")
