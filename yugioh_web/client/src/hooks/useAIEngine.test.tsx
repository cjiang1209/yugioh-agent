import { describe, it, expect, vi, afterEach } from "vitest";
import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { useAIEngine } from "./useAIEngine";
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
    recommended_action_index: null,
  };
}

function stubFetch(reward: number) {
  vi.stubGlobal(
    "fetch",
    vi.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(terminalResponse(reward)),
      })
    )
  );
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
});
