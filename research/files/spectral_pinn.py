"""
Spectral PINN with Gauss-Newton Optimization: dense solvers (v2)
=================================================================

Problems solved on x ∈ [-1, 1] with u(±1) = 0:

  Problem A (linear)    : -u'' = f
  Problem B (nonlinear) : -u'' + u³ = f

For both, u*(x) = sin(π x) is the manufactured exact solution.

Methods:
  M0  Naive            : direct lstsq (A) / scipy LM with analytic J_u (B)
  M1  Gauss-Newton SMW : closed-form K×K solve via Woodbury
  M2  Gauss-Newton LSQR: matrix-free Krylov solve

Robust enhancements vs the original draft:
  * Vectorised B = ∂c/∂θ via torch.func.jacrev (replaces K-loop).
  * Adaptive Levenberg–Marquardt damping with rejection.
  * Backtracking line search on accepted directions.
  * Cached G_lin = (A_lin Φ)ᵀ(A_lin Φ) — both for linear and as the
    fixed part of the nonlinear Gram matrix.
  * Cholesky solves with adaptive jitter (no explicit inverse).
  * Early termination on residual norm.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import least_squares

torch.set_default_dtype(torch.float64)


# =============================================================================
# 1. Chebyshev spectral machinery
# =============================================================================

def chebyshev_cgl(N: int) -> torch.Tensor:
    j = torch.arange(N + 1, dtype=torch.float64)
    return torch.cos(torch.pi * j / N)


def chebyshev_diff_matrix(N: int):
    """Return (x, D) where D is the (N+1)×(N+1) differentiation matrix."""
    x = chebyshev_cgl(N)
    c = torch.ones(N + 1, dtype=torch.float64)
    c[0] = c[-1] = 2.0

    X = x.unsqueeze(0)
    dX = X.T - X + torch.eye(N + 1)

    i_idx = torch.arange(N + 1).unsqueeze(1)
    j_idx = torch.arange(N + 1).unsqueeze(0)
    sign = (-1.0) ** (i_idx + j_idx)
    C_ratio = c.unsqueeze(1) / c.unsqueeze(0)

    D = C_ratio * sign / dX
    D = D - torch.diag(D.sum(dim=1))
    return x, D


def chebyshev_basis(x: torch.Tensor, K: int) -> torch.Tensor:
    """(N+1)×K matrix Φ[i,k] = T_k(x_i)."""
    theta = torch.arccos(x.clamp(-1 + 1e-14, 1 - 1e-14))
    k = torch.arange(K, dtype=torch.float64)
    return torch.cos(theta.unsqueeze(1) * k.unsqueeze(0))


def build_A_lin(D: torch.Tensor, bc_weight: float = 10.0) -> torch.Tensor:
    """A_lin = [ -D² ; λ e₀ᵀ ; λ e_Nᵀ ]    shape (N+3, N+1)."""
    N = D.shape[0] - 1
    A_pde = -(D @ D)
    bc = torch.zeros(2, N + 1, dtype=torch.float64)
    bc[0, 0] = bc_weight
    bc[1, -1] = bc_weight
    return torch.cat([A_pde, bc], dim=0)


# =============================================================================
# 2. Problem definitions (manufactured solutions)
# =============================================================================

def true_solution(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(torch.pi * x)


def rhs_linear(x):    return (torch.pi ** 2) * torch.sin(torch.pi * x)
def rhs_nonlinear(x):
    s = torch.sin(torch.pi * x)
    return (torch.pi ** 2) * s + s ** 3


def build_rhs(x: torch.Tensor, problem: str) -> torch.Tensor:
    f = rhs_linear(x) if problem == "linear" else rhs_nonlinear(x)
    return torch.cat([f, torch.zeros(2, dtype=torch.float64)])


def residual_F(u: torch.Tensor,
               A_lin: torch.Tensor,
               f_tilde: torch.Tensor,
               problem: str) -> torch.Tensor:
    Fu = A_lin @ u
    if problem == "nonlinear":
        Np1 = u.shape[0]
        nu = torch.zeros_like(Fu)
        nu[:Np1] = u ** 3
        Fu = Fu + nu
    return Fu - f_tilde


def linearised_operator(u: torch.Tensor, A_lin: torch.Tensor,
                        problem: str) -> torch.Tensor:
    """J_u = A_lin + diag_ext(3 u²)   (or just A_lin for the linear case)."""
    Ju = A_lin.clone()
    if problem == "nonlinear":
        Np1 = u.shape[0]
        idx = torch.arange(Np1)
        Ju[idx, idx] = Ju[idx, idx] + 3.0 * u ** 2
    return Ju


# =============================================================================
# 3. Neural network coefficient model
# =============================================================================

class CoeffNet(nn.Module):
    def __init__(self, K: int, hidden: int = 8, zero_init_output: bool = True):
        super().__init__()
        self.K = K
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, K),
        )
        # Xavier on the hidden layers, zero on the output layer so that
        # the initial coefficient vector is (close to) zero and the first
        # GN step starts from u ≡ 0. Hidden biases are zero too.
        linear_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        for m in linear_layers[:-1]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        last = linear_layers[-1]
        if zero_init_output:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:
            nn.init.xavier_uniform_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self) -> torch.Tensor:
        return self.net(torch.ones(1, 1, dtype=torch.float64)).view(-1)


def flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx:idx + n].view_as(p))
        idx += n


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_jacobian_B(model: CoeffNet) -> torch.Tensor:
    """B = ∂c/∂θ ∈ ℝ^{K×P} via vectorised torch.func.jacrev."""
    params = dict(model.named_parameters())

    def fwd(p):
        return torch.func.functional_call(model, p, (), {})

    jac_dict = torch.func.jacrev(fwd)(params)
    K = model.K
    blocks = [jac_dict[name].reshape(K, -1) for name, _ in model.named_parameters()]
    return torch.cat(blocks, dim=1)


# =============================================================================
# 4. Method 0 — Naive baseline
# =============================================================================

def run_method_0(N: int,
                 problem: str,
                 bc_weight: float = 10.0,
                 lm_max_nfev: int = 200) -> dict:
    t0 = time.perf_counter()
    x, D = chebyshev_diff_matrix(N)
    A_lin = build_A_lin(D, bc_weight=bc_weight)
    f_tilde = build_rhs(x, problem)
    A_np = A_lin.numpy()
    f_np = f_tilde.numpy()
    t1 = time.perf_counter()

    if problem == "linear":
        u_np, *_ = np.linalg.lstsq(A_np, f_np, rcond=None)
    else:
        def fun(u_np):
            u = torch.from_numpy(u_np)
            return residual_F(u, A_lin, f_tilde, problem).numpy()

        def jac(u_np):
            u = torch.from_numpy(u_np)
            return linearised_operator(u, A_lin, problem).numpy()

        u0 = np.zeros(N + 1)
        sol = least_squares(fun, u0, jac=jac, method="lm",
                            max_nfev=lm_max_nfev,
                            xtol=1e-12, ftol=1e-12, gtol=1e-12)
        u_np = sol.x

    t2 = time.perf_counter()

    u_pred = torch.from_numpy(u_np)
    u_true = true_solution(x)
    l2_err = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
    res_final = float(torch.norm(residual_F(u_pred, A_lin, f_tilde, problem)))

    return {
        "N": N, "method": "Naive", "problem": problem,
        "init_time": t1 - t0,
        "train_time": t2 - t1,
        "total_time": t2 - t0,
        "l2_error": l2_err,
        "final_residual": res_final,
        "x": x.numpy(),
        "u_pred": u_np,
        "u_true": u_true.numpy(),
    }


# =============================================================================
# 5. Shared helpers for the K-space Gram matrix and gradient
# =============================================================================

def gn_normal_pieces(u: torch.Tensor,
                     r: torch.Tensor,
                     A_lin: torch.Tensor,
                     Phi: torch.Tensor,
                     G_lin: torch.Tensor,
                     problem: str):
    """
    Return (G_K, h_K) where
        G_K = (J_u Φ)ᵀ (J_u Φ)         ∈ ℝ^{K×K}
        h_K = Φᵀ J_uᵀ r                ∈ ℝ^{K}

    For the nonlinear case we exploit the structure
        J_u = A_lin + diag_ext(3u²) = A_lin + E
    so that
        G_K = G_lin + ΦᵀA_linᵀEΦ + (ΦᵀA_linᵀEΦ)ᵀ + ΦᵀE²Φ.
    Because E is diagonal (and zero on the BC rows), each term reduces
    to operations on the top N+1 rows of Φ and A_lin.
    """
    if problem == "linear":
        h_K = Phi.T @ (A_lin.T @ r)
        return G_lin, h_K

    Np1 = u.shape[0]
    e = 3.0 * u ** 2                                # (N+1,)
    Phi_top = Phi[:Np1, :]                           # but Φ has exactly N+1 rows
    # Note: Phi has shape (N+1, K), A_lin has shape (N+3, N+1). The top
    # N+1 rows of A_lin are -D² which acts on the full grid; the bottom
    # 2 rows are BC weights.
    EPhi = e.unsqueeze(1) * Phi                     # (N+1, K)
    # A_linᵀ has shape (N+1, N+3). E "lives" only on the first N+1 columns
    # of A_linᵀ (i.e. rows of A_lin), so:
    A_top = A_lin[:Np1, :]                          # (N+1, N+1)
    AtE_Phi = A_top.T @ EPhi                         # (N+1, K)
    cross = Phi.T @ AtE_Phi                          # (K, K)
    sq = EPhi.T @ EPhi                               # (K, K)
    G_K = G_lin + cross + cross.T + sq

    h_K = Phi.T @ (A_lin.T @ r) + Phi.T @ (e * r[:Np1])
    return G_K, h_K


def cholesky_with_jitter(M: torch.Tensor, jitter0: float = 1e-12,
                         max_tries: int = 10):
    """Try to factor M + ε I with progressively larger ε until success."""
    K = M.shape[0]
    I = torch.eye(K, dtype=M.dtype)
    eps = jitter0
    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(M + eps * I)
            return L, eps
        except RuntimeError:
            eps = max(eps * 10, 1e-12)
    # Last resort: symmetric eigen-regularisation
    s, V = torch.linalg.eigh(M + eps * I)
    s = torch.clamp(s, min=eps)
    return None, (V, s, eps)


def solve_with_jitter(M: torch.Tensor, b: torch.Tensor,
                      jitter0: float = 1e-12) -> torch.Tensor:
    L, info = cholesky_with_jitter(M, jitter0=jitter0)
    if L is not None:
        if b.ndim == 1:
            return torch.cholesky_solve(b.unsqueeze(1), L).squeeze(1)
        return torch.cholesky_solve(b, L)
    V, s, _ = info
    return V @ ((V.T @ b) / s)


# =============================================================================
# 6. SMW direction solver
# =============================================================================

def smw_direction(G_K: torch.Tensor,
                  B: torch.Tensor,
                  h_K: torch.Tensor,
                  mu: float) -> torch.Tensor:
    """
    Solve  (Bᵀ G_K B + μ I) Δθ = -Bᵀ h_K   using the Woodbury identity.

    Algebra
    -------
    Let g_P = Bᵀ h_K ∈ ℝ^P. Woodbury gives
        (μ I + Bᵀ G_K B)⁻¹ = (1/μ)[I − Bᵀ M⁻¹ B],   M = μ G_K⁻¹ + B Bᵀ ∈ ℝ^{K×K}.
    Multiplying  M y = B g_P  through by G_K avoids forming G_K⁻¹:
        (μ I + G_K B Bᵀ) y = G_K B g_P,
    whose solution y is already the y we need (G_K cancels both sides).
    The final direction is
        Δθ = −(1/μ)(g_P − Bᵀ y).
    """
    K = G_K.shape[0]
    g_P = B.T @ h_K                                 # (P,)
    Bg = B @ g_P                                    # (K,)
    BBt = B @ B.T                                   # (K, K)

    M_tilde = mu * torch.eye(K, dtype=G_K.dtype) + G_K @ BBt
    rhs = G_K @ Bg

    # M_tilde may be slightly non-symmetric numerically; use plain solve.
    try:
        y = torch.linalg.solve(M_tilde, rhs)
    except RuntimeError:
        # Fall back to pinv if singular.
        y = torch.linalg.pinv(M_tilde) @ rhs

    return -(1.0 / mu) * (g_P - B.T @ y)


# =============================================================================
# 7. LSQR direction solver (matrix-free, with structured matvecs)
# =============================================================================
#
# We never materialise J = J_u Φ B. The structure
#     J x = J_u (Φ (B x))
# is applied as a sequence of three cheap matvecs. For the *linear* problem
# (J_u = A_lin) we pre-multiply once to get APhi = A_lin Φ ∈ ℝ^{(N+3)×K},
# and each forward action collapses to two matvecs of size (N+3, K) and (K, P).
# For the *nonlinear* problem J_u = A_lin + E with E diagonal on the PDE
# rows, so we form A_lin Φ once and apply E elementwise — still no (N+1)×(N+1)
# multiplications.
#
# The complexities per matvec are:
#     linear:    O(KP + (N+3)K)
#     nonlinear: O(KP + (N+1)K)
# instead of the O(N²) that a naïve J_u-application would cost.

def _Q_forward(x_in: torch.Tensor,
               APhi: torch.Tensor,
               Phi: torch.Tensor,
               B: torch.Tensor,
               e_diag: Optional[torch.Tensor],
               sqrt_mu: float) -> torch.Tensor:
    """
    y = [ J Φ B x ; √μ x ]   where J Φ x_K = A_lin Φ x_K + E Φ x_K on top.
    """
    xK = B @ x_in                                   # (K,)
    top = APhi @ xK                                 # (N+3,)
    if e_diag is not None:
        # E acts only on the top N+1 rows.
        Np1 = e_diag.shape[0]
        top = top.clone()
        top[:Np1] = top[:Np1] + e_diag * (Phi @ xK)
    return torch.cat([top, sqrt_mu * x_in])


def _Q_adjoint(y_in: torch.Tensor,
               APhi: torch.Tensor,
               Phi: torch.Tensor,
               B: torch.Tensor,
               e_diag: Optional[torch.Tensor],
               sqrt_mu: float,
               Np3: int) -> torch.Tensor:
    """z = Bᵀ Φᵀ Jᵀ y₁ + √μ y₂."""
    y1, y2 = y_in[:Np3], y_in[Np3:]
    # Jᵀ y1 = A_linᵀ y1 + Eᵀ y1_top (with E padded by zeros on BC rows).
    # Equivalent in K-space:  Φᵀ Jᵀ y1 = (A_lin Φ)ᵀ y1 + Φᵀ E y1_top.
    zK = APhi.T @ y1
    if e_diag is not None:
        Np1 = e_diag.shape[0]
        zK = zK + Phi.T @ (e_diag * y1[:Np1])
    return B.T @ zK + sqrt_mu * y2


def lsqr_direction(APhi: torch.Tensor,
                   Phi: torch.Tensor,
                   B: torch.Tensor,
                   r: torch.Tensor,
                   mu: float,
                   max_iter: int,
                   e_diag: Optional[torch.Tensor] = None,
                   Np3: Optional[int] = None,
                   atol: float = 1e-10) -> torch.Tensor:
    """LSQR on min ‖[J; √μ I] Δθ − [-r; 0]‖, J = (A_lin + E) Φ B.

    Includes a simple early stop when phi_bar (the optimal residual
    estimate) drops below `atol`.
    """
    P = B.shape[1]
    if Np3 is None:
        Np3 = APhi.shape[0]
    sqrt_mu = float(mu) ** 0.5

    b = torch.cat([-r, torch.zeros(P, dtype=r.dtype)])
    x = torch.zeros(P, dtype=r.dtype)

    u = b.clone()
    beta = torch.norm(u)
    if beta < 1e-14:
        return x
    u = u / beta

    v = _Q_adjoint(u, APhi, Phi, B, e_diag, sqrt_mu, Np3)
    alpha = torch.norm(v)
    if alpha < 1e-14:
        return x
    v = v / alpha

    w = v.clone()
    phi_bar = beta
    rho_bar = alpha
    b_norm = beta

    for _ in range(max_iter):
        u_new = _Q_forward(v, APhi, Phi, B, e_diag, sqrt_mu) - alpha * u
        beta = torch.norm(u_new)
        if beta < 1e-14:
            break
        u = u_new / beta

        v_new = _Q_adjoint(u, APhi, Phi, B, e_diag, sqrt_mu, Np3) - beta * v
        alpha = torch.norm(v_new)
        if alpha < 1e-14:
            break
        v = v_new / alpha

        rho = torch.sqrt(rho_bar ** 2 + beta ** 2)
        c_ = rho_bar / rho
        s_ = beta / rho
        theta_ = s_ * alpha
        rho_bar = -c_ * alpha
        phi_ = c_ * phi_bar
        phi_bar = s_ * phi_bar

        x = x + (phi_ / rho) * w
        w = v - (theta_ / rho) * w

        # Early stop: residual estimate small in absolute or relative sense.
        if phi_bar.item() < atol * max(b_norm.item(), 1.0):
            break

    return x


# =============================================================================
# 8. Levenberg–Marquardt outer loop (shared by M1 and M2)
# =============================================================================

@dataclass
class LMConfig:
    steps: int = 50
    mu0: float = 1e-2             # initial damping
    mu_up: float = 4.0            # increase factor on rejection
    mu_down: float = 0.5          # decrease factor on acceptance
    mu_min: float = 1e-10
    mu_max: float = 1e+8
    bc_weight: float = 10.0
    refresh_B_every: int = 1
    tol_residual: float = 1e-12
    tol_rel_decrease: float = 1e-14   # stop if loss barely changes
    max_step_norm: float = 1e3        # trust-region-ish cap on ‖Δθ‖
    verbose: bool = False


def _eval_residual_at(model_state: torch.Tensor,
                      model: CoeffNet,
                      Phi: torch.Tensor,
                      A_lin: torch.Tensor,
                      f_tilde: torch.Tensor,
                      problem: str) -> tuple:
    set_flat_params(model, model_state)
    with torch.no_grad():
        c = model()
        u = Phi @ c
        r = residual_F(u, A_lin, f_tilde, problem)
    return r, u, c


def _gn_outer_loop(model: CoeffNet,
                   Phi: torch.Tensor,
                   A_lin: torch.Tensor,
                   f_tilde: torch.Tensor,
                   G_lin: torch.Tensor,
                   APhi: torch.Tensor,
                   problem: str,
                   solver: str,
                   cfg: LMConfig,
                   lsqr_iters: int = 20):
    """
    Generic LM outer loop. `solver ∈ {'smw', 'lsqr'}` picks how Δθ is
    computed each step. Adaptive damping with full step acceptance/rejection
    (no inner line search — μ does the trust-region job).
    """
    mu = cfg.mu0
    losses = []
    cur_state = flat_params(model)
    r, u, c = _eval_residual_at(cur_state, model, Phi, A_lin, f_tilde, problem)
    cur_loss = torch.norm(r).item()
    losses.append(cur_loss)

    B = build_jacobian_B(model)
    steps_since_B = 0

    accepted = 0
    rejected = 0
    Np3 = A_lin.shape[0]

    for step in range(cfg.steps):
        if cur_loss < cfg.tol_residual:
            break

        # Build direction
        if solver == "smw":
            G_K, h_K = gn_normal_pieces(u, r, A_lin, Phi, G_lin, problem)
            delta = smw_direction(G_K, B, h_K, mu)
        elif solver == "lsqr":
            e_diag = (3.0 * u ** 2) if problem == "nonlinear" else None
            delta = lsqr_direction(APhi, Phi, B, r, mu, lsqr_iters,
                                   e_diag=e_diag, Np3=Np3)
        else:
            raise ValueError(solver)

        # Cap update magnitude (defensive).
        nrm = torch.norm(delta)
        if torch.isnan(nrm) or torch.isinf(nrm):
            mu = min(mu * cfg.mu_up, cfg.mu_max)
            rejected += 1
            continue
        if nrm > cfg.max_step_norm:
            delta = delta * (cfg.max_step_norm / nrm)

        # Try the full step.
        trial_state = cur_state + delta
        r_trial, u_trial, c_trial = _eval_residual_at(
            trial_state, model, Phi, A_lin, f_tilde, problem
        )
        trial_loss = torch.norm(r_trial).item()

        if np.isnan(trial_loss) or trial_loss >= cur_loss:
            # Reject: restore and increase damping.
            set_flat_params(model, cur_state)
            mu = min(mu * cfg.mu_up, cfg.mu_max)
            rejected += 1
            if cfg.verbose and step % 5 == 0:
                print(f"   step {step:3d} REJ  |r|={cur_loss:.3e}  μ→{mu:.1e}")
            continue

        # Accept.
        rel = (cur_loss - trial_loss) / max(cur_loss, 1e-300)
        cur_state = trial_state
        r, u, c = r_trial, u_trial, c_trial
        cur_loss = trial_loss
        losses.append(cur_loss)
        mu = max(mu * cfg.mu_down, cfg.mu_min)
        accepted += 1

        steps_since_B += 1
        if steps_since_B >= cfg.refresh_B_every:
            B = build_jacobian_B(model)
            steps_since_B = 0

        if cfg.verbose and step % 5 == 0:
            print(f"   step {step:3d} ACC  |r|={cur_loss:.3e}  μ→{mu:.1e}")

        if rel < cfg.tol_rel_decrease:
            break

    return losses, {"accepted": accepted, "rejected": rejected}


# =============================================================================
# 9. Method 1 — SMW runner
# =============================================================================

def run_method_1(N: int,
                 problem: str,
                 K: Optional[int] = None,
                 hidden: int = 8,
                 cfg: Optional[LMConfig] = None) -> dict:
    cfg = cfg or LMConfig()
    t0 = time.perf_counter()

    x, D = chebyshev_diff_matrix(N)
    A_lin = build_A_lin(D, bc_weight=cfg.bc_weight)
    if K is None:
        K = max(10, min(256, N // 3 + 1))
    Phi = chebyshev_basis(x, K)
    f_tilde = build_rhs(x, problem)

    APhi = A_lin @ Phi
    G_lin = APhi.T @ APhi

    model = CoeffNet(K, hidden=hidden)
    t1 = time.perf_counter()

    losses, stats = _gn_outer_loop(
        model, Phi, A_lin, f_tilde, G_lin, APhi, problem,
        solver="smw", cfg=cfg,
    )

    t2 = time.perf_counter()

    with torch.no_grad():
        c = model()
        u_pred = Phi @ c
        u_true = true_solution(x)
        r_final = residual_F(u_pred, A_lin, f_tilde, problem)
        l2_err = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
        res_final = float(torch.norm(r_final))

    return {
        "N": N, "method": "SMW", "problem": problem, "K": K,
        "init_time": t1 - t0,
        "train_time": t2 - t1,
        "total_time": t2 - t0,
        "l2_error": l2_err,
        "final_residual": res_final,
        "losses": losses,
        "accepted": stats["accepted"],
        "rejected": stats["rejected"],
        "x": x.numpy(),
        "u_pred": u_pred.numpy(),
        "u_true": u_true.numpy(),
    }


# =============================================================================
# 10. Method 2 — LSQR runner
# =============================================================================

def run_method_2(N: int,
                 problem: str,
                 K: Optional[int] = None,
                 hidden: int = 8,
                 lsqr_iters: int = 20,
                 cfg: Optional[LMConfig] = None) -> dict:
    cfg = cfg or LMConfig()
    t0 = time.perf_counter()

    x, D = chebyshev_diff_matrix(N)
    A_lin = build_A_lin(D, bc_weight=cfg.bc_weight)
    if K is None:
        K = max(10, min(256, N // 3 + 1))
    Phi = chebyshev_basis(x, K)
    f_tilde = build_rhs(x, problem)

    APhi = A_lin @ Phi
    G_lin = APhi.T @ APhi                  # unused by LSQR but harmless

    model = CoeffNet(K, hidden=hidden)
    t1 = time.perf_counter()

    losses, stats = _gn_outer_loop(
        model, Phi, A_lin, f_tilde, G_lin, APhi, problem,
        solver="lsqr", cfg=cfg, lsqr_iters=lsqr_iters,
    )

    t2 = time.perf_counter()

    with torch.no_grad():
        c = model()
        u_pred = Phi @ c
        u_true = true_solution(x)
        r_final = residual_F(u_pred, A_lin, f_tilde, problem)
        l2_err = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
        res_final = float(torch.norm(r_final))

    return {
        "N": N, "method": "LSQR", "problem": problem, "K": K,
        "init_time": t1 - t0,
        "train_time": t2 - t1,
        "total_time": t2 - t0,
        "l2_error": l2_err,
        "final_residual": res_final,
        "losses": losses,
        "accepted": stats["accepted"],
        "rejected": stats["rejected"],
        "x": x.numpy(),
        "u_pred": u_pred.numpy(),
        "u_true": u_true.numpy(),
    }


# =============================================================================
# 11. Adjoint consistency check
# =============================================================================

def check_adjoint(N: int = 20, problem: str = "linear") -> float:
    torch.manual_seed(0)
    x_grid, D = chebyshev_diff_matrix(N)
    A_lin = build_A_lin(D)
    K = max(10, N // 3 + 1)
    Phi = chebyshev_basis(x_grid, K)
    APhi = A_lin @ Phi
    model = CoeffNet(K)
    B = build_jacobian_B(model)
    with torch.no_grad():
        u = Phi @ model()
    e_diag = (3.0 * u ** 2) if problem == "nonlinear" else None
    P = B.shape[1]
    Np3 = A_lin.shape[0]
    mu = 1e-3
    sqrt_mu = mu ** 0.5
    xv = torch.randn(P, dtype=torch.float64)
    yv = torch.randn(Np3 + P, dtype=torch.float64)
    lhs = torch.dot(_Q_forward(xv, APhi, Phi, B, e_diag, sqrt_mu), yv)
    rhs = torch.dot(xv, _Q_adjoint(yv, APhi, Phi, B, e_diag, sqrt_mu, Np3))
    return abs(lhs.item() - rhs.item())


# =============================================================================
# 12. Smoke test
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)

    print("=== adjoint consistency ===")
    for prob in ["linear", "nonlinear"]:
        err = check_adjoint(N=20, problem=prob)
        print(f"  {prob:>9}: err={err:.3e}  ({'PASS' if err < 1e-9 else 'FAIL'})")

    cfg = LMConfig(steps=50, mu0=1e-2, verbose=False)

    for N in [40, 80, 120, 160, 200]:
        print(f"\n--- N={N} ---")
        for prob in ["linear", "nonlinear"]:
            r0 = run_method_0(N, prob)
            r1 = run_method_1(N, prob, cfg=cfg)
            r2 = run_method_2(N, prob, cfg=cfg, lsqr_iters=20)
            print(f"  {prob:>9}:")
            for r in [r0, r1, r2]:
                acc = r.get("accepted", "-")
                rej = r.get("rejected", "-")
                print(f"     {r['method']:>5}  L2={r['l2_error']:.2e}  "
                      f"|r|={r['final_residual']:.2e}  "
                      f"t={r['train_time']*1e3:6.1f} ms  "
                      f"K={r.get('K','-')}  acc/rej={acc}/{rej}")
