import { describe, it, expect, afterEach, beforeAll, vi } from "vitest";
import { cleanup, fireEvent, render } from "@testing-library/react";
import { DuelBoard, type DuelBoardProps } from "./DuelBoard";
import type {
  DuelState,
  GameCard,
  PlayerState,
} from "../../../../shared/gameTypes";
import type { EngineAction } from "../../../../shared/engineTypes";

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

const summonActions: EngineAction[] = [
  {
    index: 3,
    description: "Summon Blue-Eyes",
    card_code: 89631139,
    card_name: "Blue-Eyes White Dragon",
    category: "summon",
  },
];

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

// jsdom has no layout engine, so DuelLog's auto-scroll needs a stub. Both
// describe blocks below render DuelBoard, so the stub and cleanup live here.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn();
});
afterEach(cleanup);

describe("DuelBoard result dismissal", () => {
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

/** Mid-duel with a prompt showing: the state both layout assertions need. */
const withActions: Partial<DuelBoardProps> = {
  outcome: null,
  engineActions: summonActions,
  onEngineAction: () => {},
};

describe("DuelBoard panel layout", () => {
  it("renders LOG and ACTIONS as plain sections", () => {
    const { getByText, queryByRole } = renderBoard(withActions);

    expect(getByText("CARD DETAIL")).toBeTruthy();
    expect(getByText("LOG")).toBeTruthy();
    expect(getByText("ACTIONS (1)")).toBeTruthy();
    // The text alone would match a tabbed layout too, so the role assertions
    // are what pin these down as plain sections.
    expect(queryByRole("button", { name: "LOG" })).toBeNull();
    expect(queryByRole("button", { name: "ACTIONS (1)" })).toBeNull();
  });

  it("renders an action and a log entry simultaneously", () => {
    // makeDuelState() already seeds log: ["Opponent's LP reached 0"].
    // boardProps sets no enginePrompt, so EnginePromptRouter falls through to
    // EngineActionPanel, which renders each action's `description`.
    const { getByText, queryByText } = renderBoard(withActions);

    expect(getByText("Summon Blue-Eyes")).toBeTruthy();
    expect(getByText(/Opponent's LP reached 0/)).toBeTruthy();
    // A prompt is present, so the empty-state placeholder must not also render.
    expect(queryByText("NO ACTIONS")).toBeNull();
  });

  it("shows the NO ACTIONS placeholder when there is no prompt", () => {
    const { getByText } = renderBoard({ outcome: null, engineActions: [] });
    expect(getByText("NO ACTIONS")).toBeTruthy();
  });

  it("keeps the LOG section without an ACTIONS section when engineMode is false", () => {
    const { getByText, queryByText } = renderBoard({
      outcome: null,
      engineMode: false,
    });

    expect(getByText("LOG")).toBeTruthy();
    expect(getByText(/Opponent's LP reached 0/)).toBeTruthy();
    expect(queryByText("NO ACTIONS")).toBeNull();
    expect(queryByText(/^ACTIONS/)).toBeNull();
  });

  it("arranges the three columns as detail, field, then log/actions in DOM order", () => {
    const { getByText } = renderBoard({
      outcome: null,
      engineActions: summonActions,
      onEngineAction: () => {},
    });

    const detail = getByText("CARD DETAIL");
    // The opponent's name renders in the LifePoints HUD inside the field
    // column, so it anchors the middle column.
    const field = getByText("Opponent");
    const log = getByText("LOG");

    expect(
      detail.compareDocumentPosition(field) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy();
    expect(
      field.compareDocumentPosition(log) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy();
  });

  it("shows the selected hand card's details in the CARD DETAIL panel", () => {
    const state = makeDuelState();
    const card: GameCard = {
      id: 89631139,
      name: "Blue-Eyes White Dragon",
      type: "Normal Monster",
      frameType: "normal",
      desc: "A legendary dragon...",
      atk: 3000,
      def: 2500,
      card_images: [],
      instanceId: "hand-card-1",
    };
    state.player1.hand = [card];

    const { getByAltText, getByText } = renderBoard({ state, outcome: null });

    expect(getByText("SELECT A CARD")).toBeTruthy();

    fireEvent.click(getByAltText(card.name));

    expect(getByText(card.name)).toBeTruthy();
  });
});

describe("DuelBoard autoplay control", () => {
  /** Mid-duel with a prompt showing — the pill only makes sense there. */
  const midDuel: Partial<DuelBoardProps> = {
    outcome: null,
    engineActions: summonActions,
    onEngineAction: () => {},
  };

  /** Autoplay available — a handler is supplied — with the pill in the given
   *  state. */
  const armed = (autoplay: boolean): Partial<DuelBoardProps> => ({
    autoplay,
    onToggleAutoplay: () => {},
  });

  it("hides the pill when autoplay is unavailable", () => {
    // Duel withholds the handler when AI Assist is off, since no
    // recommendations arrive for autoplay to play.
    const { queryByRole } = renderBoard({
      ...midDuel,
      autoplay: false,
      onToggleAutoplay: undefined,
    });
    expect(queryByRole("button", { name: /AUTOPLAY/i })).toBeNull();
  });

  it("labels the pill to match the autoplay prop", () => {
    const { getByRole, rerender } = renderBoard({
      ...midDuel,
      ...armed(false),
    });
    expect(getByRole("button", { name: /AUTOPLAY: OFF/i })).toBeTruthy();

    rerender(<DuelBoard {...boardProps({ ...midDuel, ...armed(true) })} />);
    expect(getByRole("button", { name: /AUTOPLAY: ON/i })).toBeTruthy();
  });

  it("calls onToggleAutoplay when the pill is clicked", () => {
    const onToggleAutoplay = vi.fn();
    const { getByRole } = renderBoard({
      ...midDuel,
      ...armed(false),
      onToggleAutoplay,
    });

    fireEvent.click(getByRole("button", { name: /AUTOPLAY: OFF/i }));
    expect(onToggleAutoplay).toHaveBeenCalledTimes(1);
  });

  it("hides the pill once the duel is over", () => {
    // Post-duel finalize has cleared the actions and the recommendation, so
    // the toggle could only flip to ON and stall.
    const { queryByRole, getByRole } = renderBoard({
      ...midDuel,
      ...armed(true),
      outcome: "win",
    });

    fireEvent.click(getByRole("button", { name: "VIEW BOARD" }));
    expect(queryByRole("button", { name: /AUTOPLAY/i })).toBeNull();
  });

  it("keeps RESTART alongside the pill", () => {
    const { getByRole } = renderBoard({
      ...midDuel,
      ...armed(true),
      onRestart: () => {},
    });
    expect(getByRole("button", { name: /AUTOPLAY: ON/i })).toBeTruthy();
    expect(getByRole("button", { name: "RESTART" })).toBeTruthy();
  });
});

describe("DuelBoard inspector control", () => {
  it("offers the inspector pill when AI Assist supplies a toggle", () => {
    const { getByText } = renderBoard({
      outcome: null,
      onToggleInspector: () => {},
      inspectorOn: false,
    });
    expect(getByText(/INSPECT: OFF/)).toBeTruthy();
  });

  it("hides the pill when AI Assist is off", () => {
    // AI Assist off reaches the board as an omitted handler.
    const { queryByText } = renderBoard({ outcome: null });
    expect(queryByText(/INSPECT:/)).toBeNull();
  });

  it("keeps the pill after the duel ends, unlike autoplay", () => {
    const { getByText, queryByText } = renderBoard({
      outcome: "win",
      onToggleInspector: () => {},
      inspectorOn: true,
      onToggleAutoplay: () => {},
    });
    // Dismiss the result overlay first -- it fully replaces the board,
    // including the control strip, until "VIEW BOARD" is clicked.
    fireEvent.click(getByText("VIEW BOARD"));
    expect(getByText(/INSPECT: ON/)).toBeTruthy();
    expect(queryByText(/AUTOPLAY:/)).toBeNull();
  });

  it("reports clicks through the handler", () => {
    const onToggleInspector = vi.fn();
    const { getByText } = renderBoard({
      outcome: null,
      onToggleInspector,
      inspectorOn: false,
    });
    fireEvent.click(getByText(/INSPECT: OFF/));
    expect(onToggleInspector).toHaveBeenCalledOnce();
  });

  it("shows the MODEL panel and the probability column together", () => {
    // One toggle governs everything the inspector adds, so the two are
    // asserted as a pair: either half appearing without the other is the bug.
    // AI Assist keeps recommending regardless -- only the readouts are hidden.
    const inspectable = {
      outcome: null,
      onToggleInspector: () => {},
      engineActions: summonActions,
      onEngineAction: () => {},
      valueTrace: [0.5],
      // Keyed by `action.index`, and summonActions' only entry is index 3.
      actionProbs: [0, 0, 0, 0.61],
    };

    const { queryByText, rerender } = renderBoard({
      ...inspectable,
      inspectorOn: false,
    });
    expect(queryByText("MODEL")).toBeNull();
    expect(queryByText("61%")).toBeNull();

    rerender(
      <DuelBoard {...boardProps({ ...inspectable, inspectorOn: true })} />
    );
    expect(queryByText("MODEL")).not.toBeNull();
    expect(queryByText("61%")).not.toBeNull();
  });

  it("puts MODEL in the left column, directly after CARD DETAIL", () => {
    const { getByText } = renderBoard({
      outcome: null,
      onToggleInspector: () => {},
      inspectorOn: true,
      valueTrace: [0.5],
    });

    // A PanelSection's title sits in a header div inside the section, so the
    // section is the title's parent and the column is the section's parent.
    const cardDetail = getByText("CARD DETAIL").parentElement!;
    const model = getByText("MODEL").parentElement!;

    expect(model.parentElement).toBe(cardDetail.parentElement);
    expect(cardDetail.nextElementSibling).toBe(model);
    // Not the right-hand column, which owns LOG.
    expect(cardDetail.parentElement).not.toContain(getByText("LOG"));

    // The 80:20 split itself is not asserted here: jsdom's CSSOM drops the
    // `flex` shorthand, so the inline style is unreadable from a test.
  });
});
