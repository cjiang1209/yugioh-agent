// ─── AI Engine Hook ─────────────────────────────────────────────────────────
// Connects to the Python FastAPI backend (/api/web/*) and maps responses
// to the existing DuelState shape consumed by DuelBoard.

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  DuelOutcome,
  DuelState,
  GameCard,
  FieldCard,
  CardPosition,
  Phase,
  PlayerState,
} from "../../../shared/gameTypes";
import type { DeckPayload } from "../../../shared/deckTypes";
import type {
  EngineAction,
  EngineBoard,
  EngineFieldCard,
  EngineGameState,
  EngineHandCard,
  EnginePrompt,
  EngineResponse,
  PendingChainEntry,
} from "../../../shared/engineTypes";
import { API_BASE } from "../lib/apiBase";
import { EVENT_DELAY_MS } from "../lib/EventReplayMachine";
import { useEventReplay } from "./useEventReplay";

export type AIEngineStatus = "idle" | "loading" | "dueling" | "ended" | "error";

export interface UseAIEngineReturn {
  state: DuelState | null;
  /**
   * Terminal result, or null while the duel is still running.
   *
   * Assigned only once the event replay for the final step has finished, so the
   * result screen never shows over a board that is still animating: `status`
   * flips to "ended" as soon as the response arrives, well before that.
   */
  outcome: DuelOutcome | null;
  engineActions: EngineAction[];
  recommendedActionIndex: number | null;
  /** Policy probabilities for the prompt on screen, index-aligned with
   *  `engineActions`. Unlike `valueTrace` this is NOT sticky: it is keyed to
   *  one prompt's action list by position, so carrying it forward would paint
   *  stale percentages onto unrelated actions. It shares `engineActions`'
   *  lifecycle exactly. */
  actionProbs: number[] | null;
  /** One V(s) sample per prompt this duel, oldest first; the last is the
   *  current evaluation. Only grows -- a response carrying no readout leaves
   *  the last one standing -- until `reset()` clears it. Accumulates whether
   *  or not the inspector is on, so turning it on shows the full history. */
  valueTrace: number[];
  /** True while the agent seat is playing the recommender's advice itself. */
  autoplay: boolean;
  /** Flip autoplay. Switching on also plays the prompt already on screen. */
  toggleAutoplay: () => void;
  enginePrompt: EnginePrompt | null;
  visibleLog: string[];
  isReplaying: boolean;
  status: AIEngineStatus;
  error: string | null;
  reset: (
    seed?: number,
    deck0?: DeckPayload,
    deck1?: DeckPayload,
    agentPlayer?: 0 | 1
  ) => Promise<void>;
  submitAction: (actionIndex: number) => Promise<void>;
}

// ─── Mapping helpers ─────────────────────────────────────────────────────────

const CARD_IMAGE_BASE = "https://images.ygoprodeck.com/images/cards_small";
const AUTO_PASS_CATEGORIES = new Set(["pass", "no"]);

/**
 * How long an autoplayed prompt stays on screen before autoplay submits it.
 *
 * Not cosmetic: the frames replay blanks the action panel for its whole
 * animated stretch, so without a dwell the only time the action list is
 * visible is one local request round trip — it flashes and is gone before you
 * can read what the recommender was choosing between. Matched to the replay's
 * own per-event rhythm so the whole duel reads at one speed.
 */
const AUTOPLAY_DWELL_MS = EVENT_DELAY_MS;
const EMPTY_EMZ: [null, null] = [null, null];

function engineCardToGameCard(
  card: EngineHandCard | EngineFieldCard,
  side: "mine" | "opp",
  zone: string,
  seq: number
): GameCard {
  const id = card.code || 0;
  const instanceId = `${id}-${side}-${zone}-${seq}`;
  return {
    id,
    instanceId,
    name: card.name || "Unknown",
    type: mapCardType(card.type),
    frameType: card.type === "monster" ? "effect" : card.type,
    desc: "",
    atk: "attack" in card ? (card.attack ?? undefined) : undefined,
    def: "defense" in card ? (card.defense ?? undefined) : undefined,
    level: card.level ?? undefined,
    race: "",
    card_images: id
      ? [
          {
            id,
            image_url: `${CARD_IMAGE_BASE}/${id}.jpg`,
            image_url_small: `${CARD_IMAGE_BASE}/${id}.jpg`,
          },
        ]
      : [],
  };
}

/** Terminal outcome for a step response, or null while the duel is running. */
function duelOutcome(done: boolean, reward: number): DuelOutcome | null {
  if (!done) return null;
  return reward > 0 ? "win" : reward < 0 ? "loss" : "draw";
}

function mapCardType(t: string): string {
  switch (t) {
    case "monster":
      return "Effect Monster";
    case "spell":
      return "Spell Card";
    case "trap":
      return "Trap Card";
    default:
      return "Effect Monster";
  }
}

function mapPosition(pos: string): CardPosition {
  switch (pos) {
    case "ATK":
      return "ATK";
    case "DEF":
      return "DEF";
    case "FACE_DOWN_DEF":
      return "FACE_DOWN_DEF";
    case "FACE_DOWN_ATK":
      return "FACE_DOWN_ATK";
    default:
      return "ATK";
  }
}

function engineFieldCardToFieldCard(
  card: EngineFieldCard,
  side: "mine" | "opp",
  zone: string,
  seq: number
): FieldCard {
  const position = mapPosition(card.position);
  return {
    card: engineCardToGameCard(card, side, zone, seq),
    position,
    faceDown: position === "FACE_DOWN_DEF" || position === "FACE_DOWN_ATK",
  };
}

function mapPhase(phase: string): Phase {
  switch (phase) {
    case "draw":
      return "DRAW";
    case "standby":
      return "STANDBY";
    case "main1":
      return "MAIN1";
    case "battle_start":
    case "battle_step":
    case "battle":
    case "damage":
    case "damage_calc":
      return "BATTLE";
    case "main2":
      return "MAIN2";
    case "end":
      return "END";
    default:
      return "MAIN1";
  }
}

function makeFaceDownCard(
  side: "mine" | "opp",
  zone: string,
  index: number
): GameCard {
  return {
    id: 0,
    instanceId: `0-${side}-${zone}-${index}`,
    name: "Card",
    type: "Effect Monster",
    frameType: "effect",
    desc: "",
    card_images: [],
  };
}

function buildDuelState(
  board: EngineBoard,
  game_state: EngineGameState,
  log: string[]
): DuelState {
  // Player (human, always player1 / bottom)
  const playerMonsters: (FieldCard | null)[] = (
    board.player.monsters ?? []
  ).map((m, i) =>
    m ? engineFieldCardToFieldCard(m, "mine", "mzone", i) : null
  );
  const playerST: (FieldCard | null)[] = (board.player.spells_traps ?? []).map(
    (m, i) => (m ? engineFieldCardToFieldCard(m, "mine", "szone", i) : null)
  );

  const player: PlayerState = {
    id: "player1",
    name: "You",
    lifePoints: board.player.lp,
    hand: (board.player.hand ?? []).map((c, i) =>
      engineCardToGameCard(c, "mine", "hand", i)
    ),
    deck: Array.from({ length: board.player.deck_count }, (_, i) =>
      makeFaceDownCard("mine", "deck", i)
    ),
    graveyard: (board.player.graveyard ?? []).map((c, i) =>
      engineCardToGameCard(c, "mine", "grave", i)
    ),
    banished: (board.player.banished ?? []).map((c, i) =>
      engineCardToGameCard(c, "mine", "banished", i)
    ),
    extraDeck: (board.player.extra_deck ?? []).map((c, i) =>
      engineCardToGameCard(c, "mine", "extra", i)
    ),
    monsterZones: playerMonsters,
    spellTrapZones: playerST,
    fieldZone: board.player.field_zone
      ? engineFieldCardToFieldCard(board.player.field_zone, "mine", "field", 0)
      : null,
    extraMonsterZones: (board.player.extra_monster_zone ?? EMPTY_EMZ).map(
      (emz, i) =>
        emz ? engineFieldCardToFieldCard(emz, "mine", "emz", i) : null
    ),
    hasNormalSummoned: false,
    hasDrawn: false,
  };

  // Opponent (always player2 / top)
  // In open-cards mode the server sends unhidden zones + hand/extra_deck arrays.
  const opp = board.opponent;
  const oppMonsters: (FieldCard | null)[] = (opp.monsters ?? []).map((m, i) =>
    m ? engineFieldCardToFieldCard(m, "opp", "mzone", i) : null
  );
  const oppST: (FieldCard | null)[] = (opp.spells_traps ?? []).map((m, i) =>
    m ? engineFieldCardToFieldCard(m, "opp", "szone", i) : null
  );

  const oppHand = opp.hand
    ? opp.hand.map((c, i) => engineCardToGameCard(c, "opp", "hand", i))
    : Array.from({ length: opp.hand_count }, (_, i) =>
        makeFaceDownCard("opp", "hand", i)
      );

  const oppExtraDeck = opp.extra_deck
    ? opp.extra_deck.map((c, i) => engineCardToGameCard(c, "opp", "extra", i))
    : Array.from({ length: opp.extra_deck_count }, (_, i) =>
        makeFaceDownCard("opp", "extra", i)
      );

  const opponent: PlayerState = {
    id: "player2",
    name: "Opponent",
    lifePoints: opp.lp,
    hand: oppHand,
    deck: Array.from({ length: opp.deck_count }, (_, i) =>
      makeFaceDownCard("opp", "deck", i)
    ),
    graveyard: (opp.graveyard ?? []).map((c, i) =>
      engineCardToGameCard(c, "opp", "grave", i)
    ),
    banished: (opp.banished ?? []).map((c, i) =>
      engineCardToGameCard(c, "opp", "banished", i)
    ),
    extraDeck: oppExtraDeck,
    monsterZones: oppMonsters,
    spellTrapZones: oppST,
    fieldZone: opp.field_zone
      ? engineFieldCardToFieldCard(opp.field_zone, "opp", "field", 0)
      : null,
    extraMonsterZones: (opp.extra_monster_zone ?? EMPTY_EMZ).map((emz, i) =>
      emz ? engineFieldCardToFieldCard(emz, "opp", "emz", i) : null
    ),
    hasNormalSummoned: false,
    hasDrawn: false,
  };

  return {
    roomId: "engine",
    phase: mapPhase(game_state.phase),
    turnNumber: game_state.turn,
    activePlayer: game_state.is_my_turn ? "player1" : "player2",
    player1: player,
    player2: opponent,
    battleStep: null,
    log,
    pendingChain: game_state.pending_chain ?? [],
  };
}

const EMPTY_PLAYER: PlayerState = {
  id: "player1",
  name: "You",
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

const INITIAL_DUEL_STATE: DuelState = {
  roomId: "engine",
  phase: "DRAW",
  turnNumber: 1,
  activePlayer: "player1",
  player1: { ...EMPTY_PLAYER },
  player2: { ...EMPTY_PLAYER, id: "player2", name: "Opponent" },
  battleStep: null,
  log: [],
  pendingChain: [],
};

// ─── Hook ────────────────────────────────────────────────────────────────────

export function useAIEngine(
  openCards: boolean = false,
  recommend: boolean = false
): UseAIEngineReturn {
  const [state, setState] = useState<DuelState | null>(null);
  const [outcome, setOutcome] = useState<DuelOutcome | null>(null);
  const [engineActions, setEngineActions] = useState<EngineAction[]>([]);
  const [actionProbs, setActionProbs] = useState<number[] | null>(null);
  const [recommendedActionIndex, setRecommendedActionIndex] = useState<
    number | null
  >(null);
  const [valueTrace, setValueTrace] = useState<number[]>([]);
  const [autoplay, setAutoplay] = useState(false);
  // The ref, not the state, is what finalize() reads. toggleAutoplay writes it
  // synchronously so a response arriving between the click and the state
  // commit cannot observe a stale value.
  const autoplayRef = useRef(false);
  // How many /reset or /step round trips are outstanding. Nothing may start a
  // request while this is above zero — the engine backing this UI cannot run
  // two concurrent requests (SUPPORTS_CONCURRENT_SESSIONS = False).
  //
  // A count rather than a flag: reset() does not bail when busy (a teardown
  // must always win), so it can overlap a pending /step. With a boolean the
  // first of the two to settle would clear it while the other was still in
  // flight, re-opening the gate.
  const inFlightRef = useRef(0);
  // The one pending autoplay submit, if any. Both scheduling sites (finalize's
  // auto-submit and toggleAutoplay's resume kick) share it, so there is never
  // more than one autoplay timer alive: a second schedule cancels the first
  // rather than leaving two timers that both fire and submit twice.
  const autoplayTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Retired by every reset and by teardown. A request carrying an earlier epoch
  // belongs to a duel, or a hook, that no longer exists, so its response is
  // dropped rather than applied: reset() does not wait for an outstanding /step,
  // and letting that step's response land afterwards would publish the old
  // duel's board and action list against the engine's new one.
  const generationRef = useRef(0);
  const [enginePrompt, setEnginePrompt] = useState<EnginePrompt | null>(null);
  const [status, setStatus] = useState<AIEngineStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<string[]>([]);

  const {
    visibleLog: replayLog,
    currentBoard,
    currentGameState,
    isReplaying,
    startReplay,
    resetReplay,
  } = useEventReplay();

  const submitRef = useRef<(index: number) => Promise<void>>(undefined);

  const cancelScheduledAutoplay = useCallback(() => {
    if (autoplayTimerRef.current !== null) {
      clearTimeout(autoplayTimerRef.current);
      autoplayTimerRef.current = null;
    }
  }, []);

  /**
   * Show `idx` for AUTOPLAY_DWELL_MS, then submit it.
   *
   * Cancels any already-pending autoplay submit first. The dwell exists so the
   * human can read the action list, which means the human can also *click* one
   * during it; that click's response schedules a fresh submit, and without the
   * cancel the older timer would still fire afterwards and play an index
   * belonging to the prompt the click replaced.
   */
  const scheduleAutoplay = useCallback(
    (idx: number) => {
      cancelScheduledAutoplay();
      autoplayTimerRef.current = setTimeout(() => {
        autoplayTimerRef.current = null;
        // Re-checked at fire time, not only at schedule time: across a dwell
        // this long the toggle can go off, and must cancel the submit rather
        // than let it play a stale action. The in-flight check is defensive --
        // every request initiator cancels this timer before starting, so it
        // should never be pending while a request is outstanding.
        if (autoplayRef.current && inFlightRef.current === 0) {
          submitRef.current?.(idx);
        }
      }, AUTOPLAY_DWELL_MS);
    },
    [cancelScheduledAutoplay]
  );

  // Teardown is routinely reached mid-duel: RESTART under the default random
  // turn order renders CoinFlipOverlay *instead of* AIModeDuel. Two things can
  // outlive the hook -- a reply in flight, and a pending dwell timer, which
  // would POST a step and have finalize schedule the next one into a request
  // loop with no UI attached.
  useEffect(
    () => () => {
      generationRef.current += 1;
      cancelScheduledAutoplay();
    },
    [cancelScheduledAutoplay]
  );

  // During replay, update DuelState from the replay machine's intermediate board snapshots.
  // replayLog is intentionally excluded — log display uses visibleLog, not state.log.
  useEffect(() => {
    if (isReplaying && currentBoard && currentGameState) {
      setState(buildDuelState(currentBoard, currentGameState, []));
    }
  }, [isReplaying, currentBoard, currentGameState]);

  const applyResponse = useCallback(
    (resp: EngineResponse) => {
      setStatus(resp.done ? "ended" : "dueling");
      setError(null);

      // The response type says `frames` is always present, but res.json() is
      // cast to it unvalidated, so a version-skewed payload can omit it.
      const frames = resp.frames ?? [];

      const finalize = () => {
        // Update the cumulative log
        const stepEvents = frames.flatMap(f => f.events);
        const newLog = [...logRef.current, ...stepEvents];
        logRef.current = newLog;
        setState(buildDuelState(resp.board, resp.game_state, newLog));
        setEnginePrompt(resp.prompt ?? null);
        // Only here, never beside setStatus above: the closing events are
        // still replaying at that point and the result screen must not show
        // until the board has caught up.
        setOutcome(duelOutcome(resp.done, resp.reward));

        // Before the branch below, so every prompt contributes a point --
        // auto-passed and autoplayed ones are real state evaluations even
        // though the human never chose. Here rather than at response receipt
        // so the sparkline advances in step with the frame replay.
        const nextRecommendation = resp.recommendation ?? null;
        // Gated on the value, not on the recommendation: a recommender with no
        // value head still recommends, so the object arrives with `value` null
        // and there is nothing to plot.
        const nextValue = nextRecommendation?.value ?? null;
        if (nextValue != null) {
          setValueTrace(prev => [...prev, nextValue]);
        }

        // Auto-pass or show actions
        if (
          !resp.done &&
          resp.actions.length === 1 &&
          AUTO_PASS_CATEGORIES.has(resp.actions[0].category)
        ) {
          setEngineActions([]);
          setActionProbs(null);
          setEnginePrompt(null);
          setRecommendedActionIndex(null);
          setTimeout(() => submitRef.current?.(resp.actions[0].index), 0);
        } else {
          const idx = nextRecommendation?.action_index ?? null;
          setEngineActions(resp.actions);
          setActionProbs(nextRecommendation?.action_probs ?? null);
          setRecommendedActionIndex(idx);

          // Autoplay: same seam as auto-pass above, but driven by the
          // recommender. Publishing the actions first, then dwelling, is what
          // makes an autoplayed duel readable — the starred action stays up
          // long enough to see what it was chosen over. No recommendation
          // means no guess: the actions stay up for the human and autoplay
          // resumes at the next prompt that has advice.
          if (
            autoplayRef.current &&
            !resp.done &&
            idx != null &&
            resp.actions.length > 0
          ) {
            scheduleAutoplay(idx);
          }
        }
      };

      // Frames are events already known to have happened server-side; replay
      // them on screen before finalize() publishes the next prompt.
      if (frames.length > 0) {
        setEngineActions([]);
        setActionProbs(null);
        setEnginePrompt(null);
        setRecommendedActionIndex(null);
        startReplay(logRef.current, frames, finalize);
      } else {
        finalize();
      }
    },
    [startReplay, scheduleAutoplay]
  );

  const toggleAutoplay = useCallback(() => {
    const next = !autoplayRef.current;
    autoplayRef.current = next;
    setAutoplay(next);
    // Resume kick. Without it, switching autoplay on while a prompt is already
    // displayed does nothing until the human plays one action by hand. A plain
    // callback rather than an effect on [autoplay]: an effect would fire twice
    // under StrictMode, which this app does not enable today but may.
    //
    // Skipped while a request is in flight. recommendedActionIndex then still
    // describes the prompt the human just left, so submitting it would play a
    // stale action as well as breaking the one-request-at-a-time rule.
    if (
      next &&
      !isReplaying &&
      inFlightRef.current === 0 &&
      recommendedActionIndex != null
    ) {
      scheduleAutoplay(recommendedActionIndex);
    } else if (!next) {
      // Toggling off cancels a pending submit outright rather than relying on
      // the timeout's own ref check. Same outcome, but it stops a dwell-length
      // timer from sitting around after the user has said stop.
      cancelScheduledAutoplay();
    }
  }, [
    isReplaying,
    recommendedActionIndex,
    scheduleAutoplay,
    cancelScheduledAutoplay,
  ]);

  const reset = useCallback(
    async (
      seed?: number,
      deck0?: DeckPayload,
      deck1?: DeckPayload,
      agentPlayer?: 0 | 1
    ) => {
      // Bumped before the await, so a second reset supersedes this one.
      const generation = ++generationRef.current;
      setStatus("loading");
      setError(null);
      resetReplay();
      logRef.current = [];
      setState(INITIAL_DUEL_STATE);
      setOutcome(null);
      setValueTrace([]);
      setActionProbs(null);
      // Every duel starts unattended-off, whether this is a restart, a coin-flip
      // restart that remounts the hook, or a deck change.
      autoplayRef.current = false;
      setAutoplay(false);
      // A submit scheduled against the previous duel's prompt is no longer
      // valid once that duel is being torn down.
      cancelScheduledAutoplay();
      // /reset is a request round trip like /step: while it's in flight,
      // recommendedActionIndex still holds the previous duel's stale value,
      // so it must count as "in flight" too or a toggle in this window could
      // kick off a submit that races the reset itself.
      inFlightRef.current += 1;
      try {
        const body: Record<string, unknown> = {};
        if (seed !== undefined) body.seed = seed;
        if (deck0 !== undefined) body.deck0 = deck0;
        if (deck1 !== undefined) body.deck1 = deck1;
        if (openCards) body.open_cards = true;
        if (agentPlayer !== undefined) body.agent_player = agentPlayer;
        if (recommend) body.recommend = true;
        const res = await fetch(`${API_BASE}/api/web/reset`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) throw new Error(`Reset failed: ${res.status}`);
        const resp: EngineResponse = await res.json();
        if (generation !== generationRef.current) return;
        applyResponse(resp);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Reset failed");
        setStatus("error");
      } finally {
        inFlightRef.current -= 1;
      }
    },
    [applyResponse, resetReplay, openCards, recommend, cancelScheduledAutoplay]
  );

  const submitAction = useCallback(
    async (actionIndex: number) => {
      // Reachable by one ordinary click: the dwell deliberately leaves the
      // action list on screen and clickable while autoplay's own /step is in
      // flight, and /step is not quick with AI assist on -- the server runs a
      // recommender forward pass every step.
      if (inFlightRef.current > 0) return;
      // The epoch this step belongs to; an intervening reset invalidates it.
      const generation = generationRef.current;
      setError(null);
      // Any submit — the human clicking during the dwell, or autoplay's own
      // timer firing — supersedes a pending autoplay submit for the prompt
      // being left behind.
      cancelScheduledAutoplay();
      inFlightRef.current += 1;
      try {
        const res = await fetch(`${API_BASE}/api/web/step`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action_index: actionIndex }),
        });
        if (!res.ok) throw new Error(`Step failed: ${res.status}`);
        const resp: EngineResponse = await res.json();
        if (generation !== generationRef.current) return;
        applyResponse(resp);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Step failed");
        setStatus("error");
      } finally {
        inFlightRef.current -= 1;
      }
    },
    [applyResponse, cancelScheduledAutoplay]
  );

  submitRef.current = submitAction;

  const visibleLog = isReplaying ? replayLog : (state?.log ?? []);

  return {
    state,
    outcome,
    engineActions,
    recommendedActionIndex,
    actionProbs,
    valueTrace,
    autoplay,
    toggleAutoplay,
    enginePrompt,
    visibleLog,
    isReplaying,
    status,
    error,
    reset,
    submitAction,
  };
}
