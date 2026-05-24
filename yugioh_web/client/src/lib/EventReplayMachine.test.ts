import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { EventReplayMachine, type ReplayCallbacks } from "./EventReplayMachine";
import type {
  EngineBoard,
  EngineGameState,
  EventFrame,
} from "../../../shared/engineTypes";

function makeBoard(lp: number): EngineBoard {
  return {
    player: {
      hand: [],
      monsters: [null, null, null, null, null],
      spells_traps: [null, null, null, null, null],
      field_zone: null,
      graveyard: [],
      banished: [],
      extra_deck: [],
      deck_count: 40,
      lp,
    },
    opponent: {
      hand_count: 5,
      monsters: [null, null, null, null, null],
      spells_traps: [null, null, null, null, null],
      field_zone: null,
      graveyard: [],
      banished: [],
      extra_deck_count: 15,
      deck_count: 40,
      lp: 8000,
    },
  };
}

function makeGameState(turn = 1): EngineGameState {
  return { turn, phase: "main1", is_my_turn: true, chain_count: 0 };
}

function makeFrame(events: string[], lp = 8000, turn = 1): EventFrame {
  return { events, board: makeBoard(lp), game_state: makeGameState(turn) };
}

function makeCallbacks() {
  return {
    onFrameChange: vi.fn(),
    onEventReveal: vi.fn(),
    onComplete: vi.fn(),
  } satisfies ReplayCallbacks;
}

describe("EventReplayMachine", () => {
  let machine: EventReplayMachine;

  beforeEach(() => {
    vi.useFakeTimers();
    machine = new EventReplayMachine();
  });

  afterEach(() => {
    machine.dispose();
    vi.useRealTimers();
  });

  it("reveals events one by one on each tick", () => {
    const cb = makeCallbacks();
    const frame = makeFrame(["e1", "e2", "e3"]);
    machine.start([], [frame], cb);

    // Before any tick — only prior log (empty)
    expect(cb.onEventReveal).toHaveBeenLastCalledWith([]);

    vi.advanceTimersByTime(1200);
    expect(cb.onEventReveal).toHaveBeenLastCalledWith(["e1"]);

    vi.advanceTimersByTime(1200);
    expect(cb.onEventReveal).toHaveBeenLastCalledWith(["e1", "e2"]);

    vi.advanceTimersByTime(1200);
    expect(cb.onEventReveal).toHaveBeenLastCalledWith(["e1", "e2", "e3"]);
  });

  it("applies board after events, then advances to next frame", () => {
    const cb = makeCallbacks();
    const f1 = makeFrame(["a"], 8000, 1);
    const f2 = makeFrame(["b"], 7000, 2);
    machine.start([], [f1, f2], cb);

    // No board change at start — frame board is post-event, shown after events
    expect(cb.onFrameChange).not.toHaveBeenCalled();

    // Reveal "a" (300ms) — frame 1 complete → board updates to f1's snapshot
    vi.advanceTimersByTime(1200);
    expect(cb.onEventReveal).toHaveBeenLastCalledWith(["a"]);
    expect(cb.onFrameChange).toHaveBeenCalledTimes(1);
    expect(cb.onFrameChange).toHaveBeenCalledWith(f1.board, f1.game_state);

    // Inter-frame pause (150ms) → reveal "b" (300ms) → frame 2 complete → f2 board
    vi.advanceTimersByTime(600 + 1200);
    expect(cb.onFrameChange).toHaveBeenCalledTimes(2);
    expect(cb.onFrameChange).toHaveBeenLastCalledWith(f2.board, f2.game_state);
  });

  it("calls onComplete after all frames", () => {
    const cb = makeCallbacks();
    const f1 = makeFrame(["a"]);
    const f2 = makeFrame(["b"]);
    machine.start([], [f1, f2], cb);

    // Frame 1 event
    vi.advanceTimersByTime(1200);
    expect(cb.onComplete).not.toHaveBeenCalled();

    // Inter-frame pause + frame 2 event
    vi.advanceTimersByTime(600 + 1200);
    expect(cb.onComplete).toHaveBeenCalledTimes(1);
  });

  it("reset cancels timers and clears state", () => {
    const cb = makeCallbacks();
    machine.start([], [makeFrame(["a", "b"])], cb);

    vi.advanceTimersByTime(1200); // reveal "a"
    machine.reset();

    expect(machine.active).toBe(false);

    // Advance further — no more callbacks
    const revealCount = cb.onEventReveal.mock.calls.length;
    vi.advanceTimersByTime(1000);
    expect(cb.onEventReveal).toHaveBeenCalledTimes(revealCount);
    expect(cb.onComplete).not.toHaveBeenCalled();
  });

  it("dispose cancels pending timers", () => {
    const cb = makeCallbacks();
    machine.start([], [makeFrame(["a", "b"])], cb);

    machine.dispose();

    const revealCount = cb.onEventReveal.mock.calls.length;
    vi.advanceTimersByTime(1000);
    expect(cb.onEventReveal).toHaveBeenCalledTimes(revealCount);
  });

  it("start with empty frames calls onComplete immediately", () => {
    const cb = makeCallbacks();
    machine.start(["prior"], [], cb);

    expect(cb.onComplete).toHaveBeenCalledTimes(1);
    expect(cb.onEventReveal).toHaveBeenCalledWith(["prior"]);
    expect(cb.onFrameChange).not.toHaveBeenCalled();
    expect(machine.active).toBe(false);
  });

  it("visibleLog includes prior log from the start", () => {
    const cb = makeCallbacks();
    const frame = makeFrame(["e1", "e2"]);
    machine.start(["old event"], [frame], cb);

    // Synchronous: prior log visible immediately
    expect(cb.onEventReveal).toHaveBeenCalledWith(["old event"]);

    // After first tick: prior + first event
    vi.advanceTimersByTime(1200);
    expect(cb.onEventReveal).toHaveBeenLastCalledWith(["old event", "e1"]);

    // After second tick: prior + both events
    vi.advanceTimersByTime(1200);
    expect(cb.onEventReveal).toHaveBeenLastCalledWith([
      "old event",
      "e1",
      "e2",
    ]);
  });

  it("onFrameChange not called at start, only after events complete", () => {
    const cb = makeCallbacks();
    const frame = makeFrame(["e1"]);
    machine.start([], [frame], cb);

    // Board stays unchanged at start — snapshot is post-event
    expect(cb.onFrameChange).not.toHaveBeenCalled();

    // After the event is revealed, board updates
    vi.advanceTimersByTime(1200);
    expect(cb.onFrameChange).toHaveBeenCalledTimes(1);
    expect(cb.onFrameChange).toHaveBeenCalledWith(
      frame.board,
      frame.game_state
    );
  });

  it("prior log merges with completed frame events across frames", () => {
    const cb = makeCallbacks();
    const f1 = makeFrame(["a"]);
    const f2 = makeFrame(["b"]);
    machine.start(["prior"], [f1, f2], cb);

    // Reveal f1 event
    vi.advanceTimersByTime(1200);
    expect(cb.onEventReveal).toHaveBeenLastCalledWith(["prior", "a"]);

    // Inter-frame pause + reveal f2 event
    vi.advanceTimersByTime(600 + 1200);
    expect(cb.onEventReveal).toHaveBeenLastCalledWith(["prior", "a", "b"]);
  });
});
