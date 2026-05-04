"""Minimal repro: per-step LSTM micro-loop on MPS, no project imports.

Replays the (T, N, H) per-step pattern of YuGiOhNet's TBPTT branch in pure
torch — strips away the trainer/network code so the repro is filable upstream
if it crashes.

Each variant tries to bisect the failing op:
  V1: per-step LSTM, no masking, single backward
  V2: per-step LSTM + mask_hx between steps, single backward
  V3: whole-sequence LSTM (passing all T steps in one call), single backward
"""
from __future__ import annotations

import torch
import torch.nn as nn


def log(msg: str) -> None:
    print(f"[trace] {msg}", flush=True)


def mask_hx(hx, dones):
    keep = (1.0 - dones.to(torch.float32)).view(1, -1, 1)
    h, c = hx
    return (h * keep, c * keep)


def variant(
    label: str, *, use_mask: bool, whole_seq: bool,
    T: int, N: int, H: int, device: torch.device,
    cell: str = "lstm",
) -> None:
    log(f"=== {label} ===")
    torch.manual_seed(0)
    cls = nn.LSTM if cell == "lstm" else nn.GRU
    lstm = cls(input_size=H, hidden_size=H, num_layers=1).to(device)

    seq = torch.randn(T, N, H, device=device, requires_grad=True)
    h0 = torch.zeros(1, N, H, device=device)
    c0 = torch.zeros(1, N, H, device=device)
    dones = torch.zeros(T, N, device=device)
    init_hx = (h0, c0) if cell == "lstm" else h0

    if whole_seq:
        out, _ = lstm(seq, init_hx)
        log(f"  whole-seq forward: out.shape={out.shape}")
    else:
        cur_hx = init_hx
        outs = []
        for t in range(T):
            step_out, cur_hx = lstm(seq[t : t + 1], cur_hx)
            outs.append(step_out)
            if use_mask:
                cur_hx = mask_hx(cur_hx, dones[t]) if cell == "lstm" \
                    else cur_hx * (1.0 - dones[t].float()).view(1, -1, 1)
        out = torch.cat(outs, dim=0)
        log(f"  per-step forward: out.shape={out.shape}")

    torch.mps.synchronize()
    log("  forward sync OK")

    loss = out.mean()
    log("  calling backward")
    loss.backward()
    torch.mps.synchronize()
    log("  backward sync OK")


def main() -> None:
    device = torch.device("mps")

    T, N, H = 8, 1, 64

    # GRU first — confirms whether the bug is LSTM-specific or applies to all
    # per-step RNN backward on MPS.  GRU passing while LSTM crashes is the
    # signature of a kernel bug in MPS's LSTM backward, not a structural
    # autograd issue with the per-step pattern.
    variant("V0: per-step GRU, no mask", use_mask=False, whole_seq=False,
            T=T, N=N, H=H, device=device, cell="gru")

    # LSTM whole-seq — confirms the LSTM kernel itself is healthy when called
    # with a sequence in one shot.
    variant("V3: whole-seq LSTM", use_mask=False, whole_seq=True,
            T=T, N=N, H=H, device=device)

    # LSTM per-step — the smoking gun.
    variant("V1: per-step LSTM, no mask, T=8 N=1", use_mask=False, whole_seq=False,
            T=T, N=N, H=H, device=device)

    # T=2 — does even 2 micro-steps trip the assertion?
    variant("V1b: per-step LSTM, no mask, T=2 N=1", use_mask=False, whole_seq=False,
            T=2, N=N, H=H, device=device)

    # N=4 (production env count) — does batch dim matter?
    variant("V1c: per-step LSTM, no mask, T=8 N=4", use_mask=False, whole_seq=False,
            T=T, N=4, H=H, device=device)

    # Mask path — only reachable if V1 is somehow unreliable.  If V1 always
    # crashes we never hit this; left in for completeness.
    variant("V2: per-step LSTM + mask_hx", use_mask=True, whole_seq=False,
            T=T, N=N, H=H, device=device)

    log("=== all variants completed ===")


if __name__ == "__main__":
    main()
