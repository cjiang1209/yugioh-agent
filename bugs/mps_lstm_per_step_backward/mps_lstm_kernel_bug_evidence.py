"""Evidence that the per-step LSTM backward crash is an MPS kernel bug, not
user error in our code.

Each test runs the SAME pattern of ops; the only thing that varies is the
device (cpu vs mps) and the API surface (LSTM vs LSTMCell, slice vs
unsqueeze). If the bug were in our usage of the API, it would also fail on
CPU and/or with a different slicing approach. If only the MPS variants fail,
the bug is in Apple's MPS LSTM backward kernel.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def log(msg: str) -> None:
    print(f"[trace] {msg}", flush=True)


def per_step_lstm_backward(
    label: str,
    *,
    device: torch.device,
    T: int,
    N: int,
    H: int,
    slice_method: str = "slice",
) -> bool:
    """Run T per-step LSTM calls + backward.  Return True if it survived."""
    log(f"=== {label} ===")
    torch.manual_seed(0)
    lstm = nn.LSTM(input_size=H, hidden_size=H, num_layers=1).to(device)
    seq = torch.randn(T, N, H, device=device, requires_grad=True)
    hx = (torch.zeros(1, N, H, device=device), torch.zeros(1, N, H, device=device))

    outs = []
    for t in range(T):
        if slice_method == "slice":
            x_t = seq[t : t + 1]
        elif slice_method == "unsqueeze":
            x_t = seq[t].unsqueeze(0)
        elif slice_method == "select":
            x_t = seq.select(0, t).unsqueeze(0)
        else:
            raise ValueError(slice_method)
        step, hx = lstm(x_t, hx)
        outs.append(step)
    out = torch.cat(outs, dim=0)

    if device.type == "mps":
        torch.mps.synchronize()
    log(f"  forward OK out.shape={out.shape}")

    out.mean().backward()

    if device.type == "mps":
        torch.mps.synchronize()
    log("  backward OK")
    return True


def per_step_lstmcell_backward(
    label: str,
    *,
    device: torch.device,
    T: int,
    N: int,
    H: int,
) -> bool:
    """LSTMCell variant — different API surface (single-step is its native
    mode; no internal seq dim). If THIS also crashes on MPS, the bug spans
    the whole MPS LSTM family. If it works while nn.LSTM(per-step) crashes,
    the bug is specifically in the seq-dim=1 backward path of nn.LSTM."""
    log(f"=== {label} ===")
    torch.manual_seed(0)
    cell = nn.LSTMCell(input_size=H, hidden_size=H).to(device)
    seq = torch.randn(T, N, H, device=device, requires_grad=True)
    h = torch.zeros(N, H, device=device)
    c = torch.zeros(N, H, device=device)

    outs = []
    for t in range(T):
        h, c = cell(seq[t], (h, c))
        outs.append(h)
    out = torch.stack(outs, dim=0)

    if device.type == "mps":
        torch.mps.synchronize()
    log(f"  forward OK out.shape={out.shape}")

    out.mean().backward()

    if device.type == "mps":
        torch.mps.synchronize()
    log("  backward OK")
    return True


def main() -> None:
    cpu = torch.device("cpu")
    mps = torch.device("mps")

    # Evidence 1: SAME code on CPU.  If this works, the user-side autograd
    # graph is well-formed; failure on MPS is a kernel issue.
    per_step_lstm_backward(
        "CPU per-step LSTM (slice) — control",
        device=cpu,
        T=8,
        N=1,
        H=64,
        slice_method="slice",
    )

    # Evidence 2: Different slicing approach on MPS.  If both crash, the
    # crash isn't tied to a particular indexing op.
    per_step_lstm_backward(
        "MPS per-step LSTM (slice)",
        device=mps,
        T=8,
        N=1,
        H=64,
        slice_method="slice",
    )

    # If we reach this, evidence 2 didn't crash — try unsqueeze.
    per_step_lstm_backward(
        "MPS per-step LSTM (unsqueeze)",
        device=mps,
        T=8,
        N=1,
        H=64,
        slice_method="unsqueeze",
    )

    # Evidence 3: LSTMCell on MPS — different API surface, single-step is
    # native there.  Tells us whether the bug is the nn.LSTM seq-len-1
    # backward path specifically, or the entire MPS LSTM family.
    per_step_lstmcell_backward(
        "MPS per-step LSTMCell",
        device=mps,
        T=8,
        N=1,
        H=64,
    )


if __name__ == "__main__":
    main()
