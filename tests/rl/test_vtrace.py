"""Tests for V-trace advantage computation."""

from __future__ import annotations

torch = __import__("pytest").importorskip("torch")

from yugioh_rl.vtrace import compute_vtrace


def _gae_lambda1(values, rewards, dones, last_values, gamma):
    """Reference GAE with lambda=1 (equivalent to on-policy V-trace)."""
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N)
    for t in reversed(range(T)):
        next_values = last_values if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * non_terminal - values[t]
        gae = delta + gamma * non_terminal * gae
        advantages[t] = gae
    return advantages, advantages + values


def test_on_policy_matches_gae_lambda1():
    """On-policy V-trace(rho=inf, c=inf) == GAE(lambda=1)."""
    torch.manual_seed(42)
    T, N = 8, 4
    log_probs = torch.randn(T, N)
    values = torch.randn(T, N)
    rewards = torch.randn(T, N)
    dones = torch.zeros(T, N)
    dones[3, 1] = 1.0  # one episode boundary
    last_values = torch.randn(N)

    adv_vt, ret_vt = compute_vtrace(
        log_probs_old=log_probs,
        log_probs_new=log_probs,  # same = on-policy
        values=values,
        rewards=rewards,
        dones=dones,
        last_values=last_values,
        gamma=0.99,
        rho_bar=float("inf"),
        c_bar=float("inf"),
    )
    adv_gae, ret_gae = _gae_lambda1(values, rewards, dones, last_values, 0.99)

    torch.testing.assert_close(adv_vt, adv_gae, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ret_vt, ret_gae, atol=1e-5, rtol=1e-5)


def test_truncation_bounds_corrections():
    """With rho_bar=1, off-policy corrections are bounded vs rho_bar=inf."""
    torch.manual_seed(7)
    T, N = 8, 2
    log_probs_old = torch.randn(T, N)
    log_probs_new = log_probs_old + 2.0  # large policy shift
    values = torch.randn(T, N)
    rewards = torch.randn(T, N)
    dones = torch.zeros(T, N)
    last_values = torch.randn(N)

    adv_trunc, _ = compute_vtrace(
        log_probs_old,
        log_probs_new,
        values,
        rewards,
        dones,
        last_values,
        gamma=0.99,
        rho_bar=1.0,
        c_bar=1.0,
    )
    adv_untrunc, _ = compute_vtrace(
        log_probs_old,
        log_probs_new,
        values,
        rewards,
        dones,
        last_values,
        gamma=0.99,
        rho_bar=float("inf"),
        c_bar=float("inf"),
    )

    # Truncation should reduce the magnitude of corrections
    assert adv_trunc.abs().sum() < adv_untrunc.abs().sum()


def test_done_boundary_isolates_episodes():
    """Rewards after a done=1 step should not affect advantages before it."""
    T, N = 4, 1
    log_probs = torch.zeros(T, N)
    values = torch.zeros(T, N)
    last_values = torch.zeros(N)

    # Baseline: no dones, large reward at t=3
    rewards_a = torch.zeros(T, N)
    rewards_a[3, 0] = 100.0
    dones_a = torch.zeros(T, N)
    adv_a, _ = compute_vtrace(
        log_probs,
        log_probs,
        values,
        rewards_a,
        dones_a,
        last_values,
        gamma=0.99,
        rho_bar=float("inf"),
        c_bar=float("inf"),
    )

    # With done=1 at t=1: reward at t=3 should NOT affect t=0 or t=1
    dones_b = torch.zeros(T, N)
    dones_b[1, 0] = 1.0
    adv_b, _ = compute_vtrace(
        log_probs,
        log_probs,
        values,
        rewards_a,
        dones_b,
        last_values,
        gamma=0.99,
        rho_bar=float("inf"),
        c_bar=float("inf"),
    )

    # t=0 advantage should be smaller with the done barrier
    assert adv_b[0, 0].abs() < adv_a[0, 0].abs()
    # t=3 advantage should be the same (after the barrier)
    torch.testing.assert_close(adv_b[3], adv_a[3])
