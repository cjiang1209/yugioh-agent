import { describe, it, expect, afterEach, beforeAll, vi } from "vitest";
import { cleanup, fireEvent, render } from "@testing-library/react";
import { DuelBoard, type DuelBoardProps } from "./DuelBoard";
import type { DuelState, PlayerState } from "../../../../shared/gameTypes";

function makePlayer(id: string, name: string): PlayerState {
  return {
    id,
    name,
    lifePoints: 8000,
    hand: [],
    deck: [],
    graveyard: [],
    banished: [],
    extraDeck: [],
    monsterZones: [null, null, null, null, null],
    spellTrapZones: [null, null, null, null, null],
    fieldZone: null,
    extraMonsterZones: [null, null],
    hasNormalSummoned: false,
    hasDrawn: false,
  };
}

function makeDuelState(): DuelState {
  return {
    roomId: "test",
    phase: "MAIN1",
    turnNumber: 1,
    activePlayer: "player1",
    player1: makePlayer("player1", "You"),
    player2: makePlayer("player2", "Opponent"),
    battleStep: null,
    log: ["Opponent's LP reached 0"],
    pendingChain: [],
  };
}

function boardProps(overrides: Partial<DuelBoardProps> = {}): DuelBoardProps {
  return {
    state: makeDuelState(),
    mySide: "player1",
    onAction: () => {},
    engineMode: true,
    outcome: "win",
    ...overrides,
  };
}

const renderBoard = (overrides: Partial<DuelBoardProps> = {}) =>
  render(<DuelBoard {...boardProps(overrides)} />);

describe("DuelBoard result dismissal", () => {
  // jsdom has no layout engine, so DuelLog's auto-scroll needs a stub.
  beforeAll(() => {
    Element.prototype.scrollIntoView = vi.fn();
  });
  afterEach(cleanup);

  it("toggles between the result and the board via VIEW BOARD / SHOW RESULT", () => {
    const { getByText, queryByText } = renderBoard();

    expect(getByText("VICTORY")).toBeTruthy();
    expect(queryByText("SHOW RESULT")).toBeNull();

    fireEvent.click(getByText("VIEW BOARD"));
    expect(queryByText("VICTORY")).toBeNull();
    expect(getByText("SHOW RESULT")).toBeTruthy();
    expect(queryByText("Opponent")).toBeTruthy(); // the board is back

    fireEvent.click(getByText("SHOW RESULT"));
    expect(getByText("VICTORY")).toBeTruthy();
    expect(queryByText("SHOW RESULT")).toBeNull();
  });

  it("replaces RESTART with SHOW RESULT once the duel is over", () => {
    const { getByText, queryByText } = renderBoard({ onRestart: () => {} });

    fireEvent.click(getByText("VIEW BOARD"));
    // Restarting now lives in the result overlay, so the board control swaps.
    expect(queryByText("RESTART")).toBeNull();
    expect(getByText("SHOW RESULT")).toBeTruthy();
  });

  it("clears the dismissal when the next duel starts", () => {
    const { getByText, queryByText, rerender } = renderBoard();
    fireEvent.click(getByText("VIEW BOARD"));
    expect(queryByText("VICTORY")).toBeNull();

    rerender(<DuelBoard {...boardProps({ outcome: null })} />);
    rerender(<DuelBoard {...boardProps({ outcome: "loss" })} />);

    expect(getByText("DEFEAT")).toBeTruthy();
  });
});
