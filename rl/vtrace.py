"""V-trace off-policy advantage estimation.

Replaces GAE for the async actor-learner path. When ``rho_bar`` and ``c_bar``
are infinite and the behavior policy equals the target policy (zero version
lag), the computation degenerates to GAE(lambda=1).

Reference: Espeholt et al., "IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures", 2018.
"""

from __future__ import annotations

import torch


def compute_vtrace(
    log_probs_old: torch.Tensor,
    log_probs_new: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute V-trace advantages and returns.

    Args:
        log_probs_old: Log-probs under the behavior policy, shape (T, N).
        log_probs_new: Log-probs under the current policy, shape (T, N).
        values: Value estimates under the current policy, shape (T, N).
        rewards: Per-step rewards, shape (T, N).
        dones: Episode termination flags (0 or 1), shape (T, N).
        last_values: Bootstrap values at t=T, shape (N,).
        gamma: Discount factor.
        rho_bar: IS truncation threshold for the advantage weight.
        c_bar: IS truncation threshold for the trace-cutting coefficient.

    Returns:
        (advantages, returns) each of shape (T, N).
    """
    T, N = rewards.shape

    # Importance sampling ratios (per-step)
    is_ratio = (log_probs_new - log_probs_old).exp()
    rho = torch.clamp(is_ratio, max=rho_bar)
    c = torch.clamp(is_ratio, max=c_bar)

    non_terminal = 1.0 - dones

    # Backward sweep: compute v_s (V-trace corrected values)
    v_s = torch.zeros(T + 1, N, device=rewards.device)
    v_s[T] = last_values

    for t in reversed(range(T)):
        next_v = values[t + 1] if t < T - 1 else last_values
        delta = rho[t] * (rewards[t] + gamma * next_v * non_terminal[t] - values[t])
        v_s[t] = values[t] + delta + gamma * non_terminal[t] * c[t] * (v_s[t + 1] - next_v)

    # Advantages: rho * (r + gamma * v_s[t+1] - V(s_t))
    advantages = rho * (rewards + gamma * non_terminal * v_s[1:] - values)
    returns = advantages + values

    return advantages, returns
