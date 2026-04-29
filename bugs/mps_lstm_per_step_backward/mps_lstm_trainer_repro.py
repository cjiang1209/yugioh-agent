"""Faithful trainer-path repro of the MPS+LSTM crash.

Replays the EXACT update path of `_run_update_tbptt` on MPS using synthetic
buffer data shaped to the tiny crashing config (n=4, T=8, hidden=64,
minibatch_size=8, num_epochs=1). No engine, no env, just torch.

Goal: print before each suspect op so when the assertion fires we know which
op the Metal driver tripped on.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

# Project imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from yugioh_rl.config import TrainingConfig  # noqa: E402
from yugioh_rl.network import YuGiOhNet  # noqa: E402


def log(msg: str) -> None:
    print(f"[trace] {msg}", flush=True)


def main() -> None:
    device = torch.device("mps")
    torch.manual_seed(0)
    np.random.seed(0)

    # Tiny crashing config — match the prior reproduction.
    cfg = TrainingConfig(
        num_envs=4,
        deck_paths=["assets/decks/blue_eyes.ydk"],
        rollout_steps=8,
        bptt_chunk_len=8,
        num_epochs=1,
        minibatch_size=8,
        total_timesteps=256,
        rnn_type="lstm",
        rnn_hidden_dim=64,
        rnn_num_layers=1,
        device="mps",
    )

    log("building network on MPS")
    net = YuGiOhNet.from_config(cfg).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)

    T, N = cfg.rollout_steps, cfg.num_envs
    L = cfg.bptt_chunk_len
    envs_per_mb = cfg.minibatch_size // T

    log(f"shapes: T={T} N={N} L={L} envs_per_mb={envs_per_mb} hidden={cfg.rnn_hidden_dim}")

    # Synthesize a buffer minibatch on MPS, shapes match get_recurrent_batches.
    obs_cards = torch.zeros((T, envs_per_mb, 200, 42), dtype=torch.uint8, device=device)
    obs_global = torch.zeros((T, envs_per_mb, 20), dtype=torch.uint8, device=device)
    obs_actions = torch.zeros((T, envs_per_mb, 32, 12), dtype=torch.uint8, device=device)
    action_mask = torch.ones((T, envs_per_mb, 32), dtype=torch.int8, device=device)
    actions = torch.zeros((T, envs_per_mb), dtype=torch.long, device=device)
    old_log_probs = torch.zeros((T, envs_per_mb), device=device)
    advantages = torch.zeros((T, envs_per_mb), device=device)
    returns = torch.zeros((T, envs_per_mb), device=device)
    dones = torch.zeros((T, envs_per_mb), device=device)

    log("building hx_initial via init_hx + slice_hx (replays trainer path)")
    full_hx = net.init_hx(N, device)
    log(f"  full_hx[0].shape = {full_hx[0].shape}, device={full_hx[0].device}")
    env_idx_t = torch.from_numpy(np.array([0], dtype=np.int64)).long().to(device)
    log(f"  env_idx_t = {env_idx_t}, device={env_idx_t.device}, shape={env_idx_t.shape}")
    log("  calling YuGiOhNet.slice_hx ...")
    hx_initial = YuGiOhNet.slice_hx(full_hx, env_idx_t)
    log(f"  sliced hx_initial[0].shape = {hx_initial[0].shape}")

    # Force any deferred MPS ops to finalize so a later crash isn't mis-attributed.
    torch.mps.synchronize()
    log("MPS sync after slice_hx OK")

    optimizer.zero_grad()
    hx = hx_initial
    flat = L * envs_per_mb

    for chunk_start in range(0, T, L):
        chunk = slice(chunk_start, chunk_start + L)
        log(f"--- chunk start={chunk_start} ---")

        log("  calling net.forward (TBPTT path with seq_shape)")
        logits, values, hx_new = net(
            obs_cards[chunk].reshape(flat, 200, 42),
            obs_global[chunk].reshape(flat, 20),
            obs_actions[chunk].reshape(flat, 32, 12),
            action_mask[chunk].reshape(flat, 32),
            hx=hx,
            seq_shape=(L, envs_per_mb),
            dones=dones[chunk],
        )
        torch.mps.synchronize()
        log(f"  forward OK, logits.shape={logits.shape}, values.shape={values.shape}")

        log("  computing loss + backward (per-chunk)")
        # Match the trainer's loss structure (scalar combining policy/value/entropy).
        loss = (logits.mean() + values.mean()) * (L / T)
        loss.backward()
        torch.mps.synchronize()
        log("  backward OK")

        log("  calling YuGiOhNet.detach_hx")
        hx = YuGiOhNet.detach_hx(hx_new)
        torch.mps.synchronize()
        log("  detach_hx OK")

    log("calling clip_grad_norm")
    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
    torch.mps.synchronize()
    log("clip_grad_norm OK")

    log("calling optimizer.step")
    optimizer.step()
    torch.mps.synchronize()
    log("optimizer.step OK")

    log("=== full TBPTT minibatch completed without crash ===")


if __name__ == "__main__":
    main()
