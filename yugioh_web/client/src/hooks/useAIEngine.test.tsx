import { describe, it, expect, vi, afterEach } from "vitest";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { useAIEngine } from "./useAIEngine";
import { EVENT_DELAY_MS } from "../lib/EventReplayMachine";
import type { EngineBoard, EngineGameState } from "../../../shared/engineTypes";

// ─── Fixtures ────────────────────────────────────────────────────────────────

const emptySide = {
  lp: 0,
  hand_count: 0,
  deck_count: 0,
  extra_deck_count: 0,
  monsters: [null, null, null, null, null],
  spells_traps: [null, null, null, null, null],
  extra_monster_zone: [null, null],
  field_zone: null,
  graveyard: [],
  banished: [],
};

const board = {
  player: { ...emptySide },
  opponent: { ...emptySide },
} as unknown as EngineBoard;

const gameState = {
  phase: "main1",
  turn: 3,
  is_my_turn: true,
  pending_chain: [],
} as unknown as EngineGameState;

/** Terminal response with two frames of two events — several non-final timers. */
function terminalResponse(reward: number) {
  return {
    board,
    game_state: gameState,
    actions: [],
    prompt: null,
    done: true,
    reward,
    frames: [
      {
        events: ["attack declared", "damage dealt"],
        board,
        game_state: gameState,
      },
      { events: ["LP reached 0", "duel over"], board, game_state: gameState },
    ],
    recommendation: null,
  };
}

type FetchResult = { ok: boolean; json: () => Promise<unknown> };

/** The shape `fetch` resolves to, as much of it as this hook reads. */
function okJson(body: unknown): FetchResult {
  return { ok: true, json: () => Promise.resolve(body) };
}

/** Answers every request with `body`. */
function stubFetchBody(body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(() => Promise.resolve(okJson(body)))
  );
}

function stubFetch(reward: number) {
  stubFetchBody(terminalResponse(reward));
}

const summonAction = {
  index: 3,
  description: "Summon Blue-Eyes",
  card_code: 89631139,
  card_name: "Blue-Eyes White Dragon",
  category: "summon",
};

/** Non-terminal response offering one action. `frames: []` keeps replay out of
 *  the picture so finalize() runs synchronously. */
function promptResponse(recommended: number | null, index = 3) {
  return {
    board,
    game_state: gameState,
    actions: [{ ...summonAction, index }],
    prompt: null,
    done: false,
    reward: 0,
    frames: [],
    recommendation:
      recommended == null
        ? null
        : { action_index: recommended, value: null, action_probs: null },
  };
}

/** Serves `responses` in order (the last repeats for any further calls) and
 *  records the action_index of every /step call, in order. */
function stubFetchSequence(responses: unknown[]): number[] {
  const submitted: number[] = [];
  let i = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string, init?: RequestInit) => {
      if (url.endsWith("/api/web/step")) {
        submitted.push(JSON.parse(String(init?.body)).action_index as number);
      }
      const body = responses[Math.min(i, responses.length - 1)];
      i += 1;
      return Promise.resolve(okJson(body));
    })
  );
  return submitted;
}

/** Serves `resetBody` for /reset immediately, but leaves every /step request
 *  pending until `resolveStep` is called — so a request can be held genuinely
 *  in flight while other things happen. Records the action_index of every /step
 *  call, in order, including any unwanted concurrent ones. */
function stubFetchWithControlledStep(resetBody: unknown): {
  submitted: number[];
  resolveStep: (body: unknown) => void;
} {
  const submitted: number[] = [];
  let resolveStep!: (body: unknown) => void;
  const stepPending = new Promise<FetchResult>(resolve => {
    resolveStep = body => resolve(okJson(body));
  });
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string, init?: RequestInit) => {
      if (url.endsWith("/api/web/step")) {
        submitted.push(JSON.parse(String(init?.body)).action_index as number);
        return stepPending;
      }
      return Promise.resolve(okJson(resetBody));
    })
  );
  return { submitted, resolveStep };
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe("useAIEngine outcome", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
    cleanup();
  });

  it("stays null through the closing replay, then reports the win", async () => {
    vi.useFakeTimers();
    stubFetch(1);
    const { result } = renderHook(() => useAIEngine());

    await act(async () => {
      await result.current.reset();
    });

    // The response has arrived: status flips immediately, but the board is
    // still replaying the closing events — no result yet.
    expect(result.current.status).toBe("ended");
    expect(result.current.isReplaying).toBe(true);
    expect(result.current.outcome).toBeNull();

    // Step a couple of non-final timers; the replay must still be in flight.
    await act(async () => {
      await vi.advanceTimersToNextTimerAsync();
    });
    expect(result.current.isReplaying).toBe(true);
    expect(result.current.outcome).toBeNull();
    await act(async () => {
      await vi.advanceTimersToNextTimerAsync();
    });
    expect(result.current.isReplaying).toBe(true);
    expect(result.current.outcome).toBeNull();

    // Drain the rest — the last timer completes the replay and finalizes.
    await act(async () => {
      await vi.runAllTimersAsync();
    });
    expect(result.current.isReplaying).toBe(false);
    expect(result.current.outcome).toBe("win");
  });

  it.each([
    [1, "win"],
    [-1, "loss"],
    [0, "draw"],
  ] as const)("maps reward %i to %s", async (reward, expected) => {
    vi.useFakeTimers();
    stubFetch(reward);
    const { result } = renderHook(() => useAIEngine());

    await act(async () => {
      await result.current.reset();
    });
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(result.current.outcome).toBe(expected);
  });

  it("clears the outcome when a new duel is reset", async () => {
    stubFetch(1);
    const { result } = renderHook(() => useAIEngine());

    vi.useFakeTimers();
    await act(async () => {
      await result.current.reset();
    });
    await act(async () => {
      await vi.runAllTimersAsync();
    });
    expect(result.current.outcome).toBe("win");

    // A restart mid-flight must not leave the previous result on screen.
    vi.useRealTimers();
    act(() => {
      void result.current.reset();
    });
    await waitFor(() => expect(result.current.outcome).toBeNull());
  });

  it("finalizes without throwing when the response carries no frames", async () => {
    // A version-skewed or malformed backend can omit `frames`. The board must
    // still update rather than dying inside finalize().
    stubFetchBody({ ...terminalResponse(1), frames: undefined });
    const { result } = renderHook(() => useAIEngine());

    await act(async () => {
      await result.current.reset();
    });

    expect(result.current.error).toBeNull();
    expect(result.current.status).toBe("ended");
    expect(result.current.outcome).toBe("win");
  });
});

describe("useAIEngine autoplay", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
    cleanup();
  });

  it("plays the recommended action when autoplay is switched on", async () => {
    vi.useFakeTimers();
    const submitted = stubFetchSequence([
      promptResponse(3),
      terminalResponse(1),
    ]);
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset();
    });
    expect(submitted).toEqual([]);

    // Switching on must act on the prompt already displayed, not wait for the
    // next one.
    act(() => {
      result.current.toggleAutoplay();
    });
    expect(result.current.autoplay).toBe(true);
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(submitted).toEqual([3]);
  });

  it("keeps playing until the duel ends", async () => {
    vi.useFakeTimers();
    const submitted = stubFetchSequence([
      promptResponse(3, 3),
      promptResponse(4, 4),
      terminalResponse(1),
    ]);
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset();
    });
    act(() => {
      result.current.toggleAutoplay();
    });
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    // reset served the first prompt; autoplay played 3, then 4, then received
    // the terminal response and stopped.
    expect(submitted).toEqual([3, 4]);
    expect(result.current.outcome).toBe("win");
  });

  it("submits nothing while autoplay is off", async () => {
    vi.useFakeTimers();
    const submitted = stubFetchSequence([promptResponse(3)]);
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset();
    });
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(result.current.autoplay).toBe(false);
    expect(submitted).toEqual([]);
  });

  it("hands control back, staying on, when there is no recommendation", async () => {
    vi.useFakeTimers();
    // The advised prompt comes first so autoplay plays it and is already ON
    // when the unadvised one arrives — that is the case finalize's own null
    // guard covers, and it is unreachable if autoplay is off to begin with.
    const submitted = stubFetchSequence([
      promptResponse(3),
      promptResponse(null, 8),
    ]);
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset();
    });
    act(() => {
      result.current.toggleAutoplay();
    });
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    // Played the advised prompt, then stopped rather than guessing at the
    // unadvised one, leaving its actions up for the human.
    expect(submitted).toEqual([3]);
    expect(result.current.engineActions).toHaveLength(1);
    // Still armed: the next prompt that does carry advice resumes autoplay.
    expect(result.current.autoplay).toBe(true);
  });

  it("clears autoplay when a new duel is reset", async () => {
    vi.useFakeTimers();
    stubFetchBody(promptResponse(null));
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset();
    });
    act(() => {
      result.current.toggleAutoplay();
    });
    expect(result.current.autoplay).toBe(true);

    await act(async () => {
      await result.current.reset();
    });
    expect(result.current.autoplay).toBe(false);
  });

  it("cancels the resume-kick submit when toggled off before the timer fires", async () => {
    vi.useFakeTimers();
    const submitted = stubFetchSequence([promptResponse(3)]);
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset();
    });

    // Turning autoplay on with a recommendation already displayed schedules a
    // resume-kick submit. Toggling off again before that timer fires must
    // cancel it, not let it play a stale action.
    act(() => {
      result.current.toggleAutoplay();
      result.current.toggleAutoplay();
    });
    expect(result.current.autoplay).toBe(false);
    // The timer must be gone, not merely disarmed: the callback's own ref check
    // would suppress the submit either way, so counting timers is what pins the
    // cancellation rather than the suppression.
    expect(vi.getTimerCount()).toBe(0);

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(submitted).toEqual([]);
  });

  it("cancels finalize's auto-submit when toggled off before the timer fires", async () => {
    vi.useFakeTimers();
    const submitted = stubFetchSequence([
      promptResponse(null),
      promptResponse(3, 3),
    ]);
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset();
    });
    // Arm autoplay while there is no recommendation on screen yet, so this
    // toggle itself has nothing to resume-kick.
    act(() => {
      result.current.toggleAutoplay();
    });
    expect(result.current.autoplay).toBe(true);

    // A manual submit (standing in for a human click) brings back a
    // recommendation; finalize() schedules its own auto-submit for it.
    await act(async () => {
      await result.current.submitAction(5);
    });

    // Toggle off before that timer fires.
    act(() => {
      result.current.toggleAutoplay();
    });
    expect(result.current.autoplay).toBe(false);
    // As above: the timer must be gone, not just disarmed by the ref check.
    expect(vi.getTimerCount()).toBe(0);

    await act(async () => {
      await vi.runAllTimersAsync();
    });

    // Only the manual submit landed; finalize's scheduled follow-up did not.
    expect(submitted).toEqual([5]);
  });

  it("does not issue a second /step when autoplay is switched on while one is in flight", async () => {
    vi.useFakeTimers();
    const { submitted, resolveStep } = stubFetchWithControlledStep(
      promptResponse(3, 3)
    );
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset();
    });
    expect(submitted).toEqual([]);

    // The human clicks the displayed action. Its /step response is held
    // pending deliberately, so the request stays genuinely in flight.
    act(() => {
      void result.current.submitAction(3);
    });

    // Switching autoplay on now must not schedule a submit at all: the request
    // in flight means recommendedActionIndex still describes the prompt the
    // human just left, and the backend cannot run two sessions concurrently.
    act(() => {
      result.current.toggleAutoplay();
    });
    // No timer, so the guard has to be the kick's own in-flight check. Letting
    // a timer be scheduled and relying on its callback to bail would pass even
    // with that check removed, because the callback re-checks too.
    expect(vi.getTimerCount()).toBe(0);

    // Resolve the step to a terminal response, which schedules nothing further,
    // then drain the closing replay.
    await act(async () => {
      resolveStep(terminalResponse(1));
      await vi.runAllTimersAsync();
    });

    expect(submitted).toEqual([3]);
  });

  it("auto-submits a lone pass action even when autoplay is off", async () => {
    // The auto-pass branch in finalize() is unconditional — it must fire
    // regardless of the autoplay toggle. A future refactor that folded it
    // under the autoplay gate should fail this test loudly.
    vi.useFakeTimers();
    const passOnlyResponse = {
      ...promptResponse(null, 7),
      actions: [{ ...summonAction, index: 7, category: "pass" }],
    };
    const submitted = stubFetchSequence([
      passOnlyResponse,
      terminalResponse(1),
    ]);
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset();
    });
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(result.current.autoplay).toBe(false);
    expect(submitted).toEqual([7]);
  });

  it("leaves the action list up for the dwell before submitting", async () => {
    vi.useFakeTimers();
    const submitted = stubFetchSequence([
      promptResponse(3),
      terminalResponse(1),
    ]);
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset();
    });
    act(() => {
      result.current.toggleAutoplay();
    });

    // Most of the dwell has elapsed and the actions are still on screen: this
    // is the window that makes an autoplayed duel readable.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(EVENT_DELAY_MS - 1);
    });
    expect(submitted).toEqual([]);
    expect(result.current.engineActions).toHaveLength(1);
    expect(result.current.recommendedActionIndex).toBe(3);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });
    expect(submitted).toEqual([3]);
  });

  it("does not submit a stale index when the human clicks during the dwell", async () => {
    vi.useFakeTimers();
    // /reset offers a prompt recommending 3; every /step stays pending until
    // we resolve it, so the click and the dwell overlap deterministically.
    const { submitted, resolveStep } = stubFetchWithControlledStep(
      promptResponse(3)
    );
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset();
    });
    act(() => {
      result.current.toggleAutoplay();
    });

    // Part-way through the dwell the human picks a different action.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200);
    });
    act(() => {
      void result.current.submitAction(9);
    });

    // Land that response *before* the original dwell deadline, so inFlightRef
    // is back to false when it arrives. The timeout's own ref checks therefore
    // cannot save us here — only cancelling the superseded timer can.
    await act(async () => {
      resolveStep(terminalResponse(1));
      await vi.advanceTimersByTimeAsync(100);
    });
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    // Only the human's action. Index 3 belonged to the prompt they replaced,
    // and the duel is over by the time the old timer would have fired.
    expect(submitted).toEqual([9]);
  });

  it("issues no request after unmount with a dwell still pending", async () => {
    vi.useFakeTimers();
    const submitted = stubFetchSequence([
      promptResponse(3),
      terminalResponse(1),
    ]);
    const { result, unmount } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset();
    });
    act(() => {
      result.current.toggleAutoplay();
    });

    // RESTART under the default random turn order swaps in CoinFlipOverlay and
    // unmounts the hook mid-dwell. The pending timer must not survive to POST
    // /step — if it did, finalize would schedule the next one and the loop
    // would run on against the duel the remount is about to start.
    unmount();
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(submitted).toEqual([]);
  });

  it("ignores a response that lands after unmount", async () => {
    vi.useFakeTimers();
    // The held step resolves to a pass-only prompt, which finalize auto-submits
    // regardless of autoplay. So if a post-unmount response were applied at
    // all, it would POST a step against the duel the remount is starting.
    const passOnly = {
      ...promptResponse(null, 7),
      actions: [{ ...summonAction, index: 7, category: "pass" }],
    };
    const { submitted, resolveStep } = stubFetchWithControlledStep(
      promptResponse(3)
    );
    const { result, unmount } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset();
    });
    act(() => {
      void result.current.submitAction(4);
    });
    expect(submitted).toEqual([4]);

    // A restart that animates the coin flip swaps this hook out while its
    // request is still in flight.
    unmount();
    await act(async () => {
      resolveStep(passOnly);
      await vi.runAllTimersAsync();
    });

    expect(submitted).toEqual([4]);
  });

  it("drops a step response that a reset has already superseded", async () => {
    vi.useFakeTimers();
    // /reset answers with a prompt offering action 3; the held /step would
    // answer with action 99. Whichever list ends up on screen identifies which
    // duel the hook thinks it is playing.
    const { submitted, resolveStep } = stubFetchWithControlledStep(
      promptResponse(3)
    );
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset();
    });
    act(() => {
      void result.current.submitAction(3);
    });
    expect(submitted).toEqual([3]);

    // Restarting does not wait for the outstanding step.
    await act(async () => {
      await result.current.reset();
    });
    expect(result.current.engineActions[0].index).toBe(3);

    // The superseded step now answers. Applying it would replace the fresh
    // duel's action list with the old duel's, and drop the fresh duel's
    // finalize by re-arming the replay machine.
    await act(async () => {
      resolveStep(promptResponse(null, 99));
      await vi.runAllTimersAsync();
    });

    expect(result.current.engineActions[0].index).toBe(3);
  });

  it("refuses a second request while one is already in flight", async () => {
    vi.useFakeTimers();
    const { submitted, resolveStep } = stubFetchWithControlledStep(
      promptResponse(3)
    );
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset();
    });

    // The action list stays on screen and clickable for the whole round trip,
    // so a second click is an ordinary thing to do — and must be dropped.
    act(() => {
      void result.current.submitAction(4);
    });
    act(() => {
      void result.current.submitAction(5);
    });

    expect(submitted).toEqual([4]);

    await act(async () => {
      resolveStep(terminalResponse(1));
      await vi.runAllTimersAsync();
    });
    expect(submitted).toEqual([4]);
  });
});

describe("useAIEngine inspection", () => {
  /** A non-terminal response carrying one prompt and an optional readout. */
  function inspectableResponse(recommendation: unknown) {
    return {
      board,
      game_state: gameState,
      actions: [
        {
          index: 0,
          description: "Summon",
          card_code: 0,
          card_name: "x",
          category: "summon",
        },
      ],
      prompt: null,
      done: false,
      reward: 0,
      frames: [],
      recommendation:
        recommendation == null
          ? null
          : { action_index: 0, ...(recommendation as object) },
    };
  }

  afterEach(() => {
    vi.unstubAllGlobals();
    cleanup();
  });

  it("appends the value to the trace", async () => {
    stubFetchBody(inspectableResponse({ value: 0.25, action_probs: [1] }));
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset(1);
    });

    await waitFor(() => {
      expect(result.current.valueTrace).toEqual([0.25]);
    });
  });

  it("accumulates one trace point per prompt", async () => {
    stubFetchBody(inspectableResponse({ value: 0.25, action_probs: [1] }));
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset(1);
    });
    await act(async () => {
      await result.current.submitAction(0);
    });

    await waitFor(() => {
      expect(result.current.valueTrace).toEqual([0.25, 0.25]);
    });
  });

  it("leaves the trace empty when the recommender has no value head", async () => {
    stubFetchBody(inspectableResponse(null));
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset(1);
    });

    await waitFor(() => {
      expect(result.current.valueTrace).toEqual([]);
    });
  });

  it("clears the trace on reset so a new duel starts from scratch", async () => {
    stubFetchBody(inspectableResponse({ value: 0.25, action_probs: [1] }));
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset(1);
    });
    await waitFor(() => expect(result.current.valueTrace).toEqual([0.25]));

    await act(async () => {
      await result.current.reset(2);
    });
    await waitFor(() => expect(result.current.valueTrace).toEqual([0.25]));
  });

  it("counts an auto-passed prompt's readout in the trace", async () => {
    // The trace push in finalize() runs before the auto-pass/else split, so a
    // pass-only prompt -- which never reaches the branch the other tests in
    // this file exercise -- must still land its value here.
    vi.useFakeTimers();
    const passResponse = {
      ...inspectableResponse({ value: 0.5, action_probs: [1] }),
      actions: [
        {
          index: 7,
          description: "Pass",
          card_code: 0,
          card_name: "x",
          category: "pass",
        },
      ],
    };
    const submitted = stubFetchSequence([passResponse, terminalResponse(1)]);
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset(1);
    });
    await act(async () => {
      await vi.runAllTimersAsync();
    });

    expect(submitted).toEqual([7]);
    expect(result.current.valueTrace).toEqual([0.5]);
    vi.useRealTimers();
  });

  it("drops the per-action probabilities when a response carries none", async () => {
    // V(s) is sticky, the probabilities are not: they are keyed to one prompt's
    // action list by position. Inference can fail mid-duel (the server catches
    // it and sends recommendation: null with a fresh actions list), and carrying the
    // previous array forward would paint its percentages onto unrelated
    // actions -- silently, and wrongly.
    stubFetchSequence([
      inspectableResponse({ value: 0.25, action_probs: [0.6, 0.4] }),
      { ...inspectableResponse(null), recommendation: null },
    ]);
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset(1);
    });
    expect(result.current.actionProbs).toEqual([0.6, 0.4]);

    await act(async () => {
      await result.current.submitAction(0);
    });

    await waitFor(() => {
      expect(result.current.engineActions.length).toBeGreaterThan(0);
      expect(result.current.actionProbs).toBeNull();
    });
    // The history still holds -- only the position-keyed array is dropped.
    expect(result.current.valueTrace).toEqual([0.25]);
  });

  it("holds the last readout when a response carries none", async () => {
    // A terminal step sends recommendation: null. Blanking on it would wipe the
    // number at exactly the moment you want to read the duel back, so the
    // panel keeps the newest evaluation it has seen until the next reset.
    vi.useFakeTimers();
    stubFetchSequence([
      inspectableResponse({ value: 0.25, action_probs: [1] }),
      terminalResponse(1),
    ]);
    const { result } = renderHook(() => useAIEngine(false, true));

    await act(async () => {
      await result.current.reset(1);
    });
    expect(result.current.valueTrace).toEqual([0.25]);

    await act(async () => {
      await result.current.submitAction(0);
    });
    // Drain the closing replay. `status` flips to "ended" the moment the
    // response lands, before finalize() runs, so waiting on it would assert
    // nothing -- `outcome` is set inside finalize and is the real signal.
    await act(async () => {
      await vi.runAllTimersAsync();
    });
    expect(result.current.outcome).toBe("win");

    // The terminal response carried no readout, so it added no trace point.
    expect(result.current.valueTrace).toEqual([0.25]);
    vi.useRealTimers();
  });

  it("keeps the readout on screen for an auto-passed prompt", async () => {
    // Most prompts in a real duel are single-action passes, so blanking the
    // readout here would leave V(s) absent while the sparkline and the prompt
    // count kept advancing. Only the recommendation clears -- there is nothing
    // to click -- and the value describes the board either way.
    vi.useFakeTimers();
    const passResponse = {
      ...inspectableResponse({ value: 0.5, action_probs: [1] }),
      actions: [
        {
          index: 7,
          description: "Pass",
          card_code: 0,
          card_name: "x",
          category: "pass",
        },
      ],
    };
    stubFetchSequence([passResponse, terminalResponse(1)]);
    const { result } = renderHook(() => useAIEngine(false, true));

    // Read between finalize() and the queued auto-submit: this is the window
    // the human actually looks at while the pass goes through.
    await act(async () => {
      await result.current.reset(1);
    });

    expect(result.current.recommendedActionIndex).toBeNull();
    expect(result.current.valueTrace).toEqual([0.5]);

    await act(async () => {
      await vi.runAllTimersAsync();
    });
    vi.useRealTimers();
  });
});
