"""Shared obs-dict -> forward-kwargs plumbing for policy action selection.

Every action-selection path (sync/async rollout collection, serving) needs to
turn an observation dict of numpy arrays into the keyword arguments for
``YuGiOhNet.forward``. That conversion used to be copy-pasted across three call
sites, so each new observation field had to be hand-wired into every copy — and
one copy silently got missed (the obs_chain train/serve bug). Centralizing it
here means a new field wires in exactly one place.
"""

from __future__ import annotations

import torch


def build_forward_inputs(
    obs,
    *,
    device=None,
    add_batch_dim: bool = False,
    guard_optional: bool = False,
) -> dict[str, torch.Tensor | None]:
    """Build the single-step ``YuGiOhNet.forward`` kwargs from an obs dict.

    Args:
        obs: mapping of observation name -> numpy array.
        device: if set, ``.to(device)`` each tensor (serving / sync collection);
            ``None`` keeps them on CPU (async workers).
        add_batch_dim: ``unsqueeze(0)`` each tensor for single-env callers;
            ``False`` when ``obs`` is already batched ``(N, ...)`` as in
            ``SubprocVecEnv`` collection.
        guard_optional: read ``event_history`` / ``pending_chain`` via ``.get()``
            and yield ``None`` when absent (serving obs dicts are built by varied
            producers); ``False`` indexes directly (collection, where
            ``TrainingEnv``/``_obs_to_numpy`` always populates the key).

    Returns:
        A dict of ``forward`` kwargs: ``obs_cards``, ``obs_global``,
        ``obs_actions``, ``action_mask``, ``obs_chain``, ``obs_event``.
        Splat it into the network call: ``net(**inputs, hx=hx)``.
    """

    def _t(arr):
        t = torch.from_numpy(arr)
        if add_batch_dim:
            t = t.unsqueeze(0)
        return t.to(device)  # .to(None) is a no-op, so no device guard needed

    def _opt(key):
        # Strict mode (collection): index directly so a missing/None value fails
        # loudly — the key is contractually always a real array there. Guarded
        # mode (serving): tolerate absence, since obs producers vary (e.g. the
        # MUD obs builder omits event_history/pending_chain entirely).
        if not guard_optional:
            return _t(obs[key])
        arr = obs.get(key)
        return None if arr is None else _t(arr)

    return {
        "obs_cards": _t(obs["cards"]),
        "obs_global": _t(obs["global_state"]),
        "obs_actions": _t(obs["actions"]),
        "action_mask": _t(obs["action_mask"]),
        "obs_chain": _opt("pending_chain"),
        "obs_event": _opt("event_history"),
    }
