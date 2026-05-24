// ─── AI Engine Hook ─────────────────────────────────────────────────────────
// Connects to the Python FastAPI backend (/api/web/*) and maps responses
// to the existing DuelState shape consumed by DuelBoard.

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  DuelState,
  GameCard,
  FieldCard,
  CardPosition,
  Phase,
  PlayerState,
  PlayerSide,
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
} from "../../../shared/engineTypes";
import { useEventReplay } from "./useEventReplay";

export type AIEngineStatus = "idle" | "loading" | "dueling" | "ended" | "error";

export interface UseAIEngineReturn {
  state: DuelState | null;
  engineActions: EngineAction[];
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
  done: boolean,
  reward: number,
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
    winner: done
      ? reward > 0
        ? "player1"
        : reward < 0
          ? "player2"
          : null
      : null,
    battleStep: null,
    log,
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
  winner: null,
  battleStep: null,
  log: [],
};

// ─── Hook ────────────────────────────────────────────────────────────────────

export function useAIEngine(
  apiUrl: string = "http://localhost:8000",
  openCards: boolean = false
): UseAIEngineReturn {
  const [state, setState] = useState<DuelState | null>(null);
  const [engineActions, setEngineActions] = useState<EngineAction[]>([]);
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

  // During replay, update DuelState from the replay machine's intermediate board snapshots.
  // replayLog is intentionally excluded — log display uses visibleLog, not state.log.
  useEffect(() => {
    if (isReplaying && currentBoard && currentGameState) {
      setState(buildDuelState(currentBoard, currentGameState, false, 0, []));
    }
  }, [isReplaying, currentBoard, currentGameState]);

  const applyResponse = useCallback(
    (resp: EngineResponse) => {
      setStatus(resp.done ? "ended" : "dueling");
      setError(null);

      const finalize = () => {
        // Update the cumulative log
        const newLog = [...logRef.current, ...resp.event_log];
        logRef.current = newLog;
        setState(
          buildDuelState(
            resp.board,
            resp.game_state,
            resp.done,
            resp.reward,
            newLog
          )
        );
        setEnginePrompt(resp.prompt ?? null);

        // Auto-pass or show actions
        if (
          !resp.done &&
          resp.actions.length === 1 &&
          AUTO_PASS_CATEGORIES.has(resp.actions[0].category)
        ) {
          setEngineActions([]);
          setEnginePrompt(null);
          setTimeout(() => submitRef.current?.(resp.actions[0].index), 0);
        } else {
          setEngineActions(resp.actions);
        }
      };

      // If frames are available, replay them before finalizing
      if (resp.frames && resp.frames.length > 0) {
        setEngineActions([]);
        setEnginePrompt(null);
        startReplay(logRef.current, resp.frames, finalize);
      } else {
        finalize();
      }
    },
    [startReplay]
  );

  const reset = useCallback(
    async (
      seed?: number,
      deck0?: DeckPayload,
      deck1?: DeckPayload,
      agentPlayer?: 0 | 1
    ) => {
      setStatus("loading");
      setError(null);
      resetReplay();
      logRef.current = [];
      setState(INITIAL_DUEL_STATE);
      try {
        const body: Record<string, unknown> = {};
        if (seed !== undefined) body.seed = seed;
        if (deck0 !== undefined) body.deck0 = deck0;
        if (deck1 !== undefined) body.deck1 = deck1;
        if (openCards) body.open_cards = true;
        if (agentPlayer !== undefined) body.agent_player = agentPlayer;
        const res = await fetch(`${apiUrl}/api/web/reset`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res.ok) throw new Error(`Reset failed: ${res.status}`);
        const resp: EngineResponse = await res.json();
        applyResponse(resp);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Reset failed");
        setStatus("error");
      }
    },
    [apiUrl, applyResponse, resetReplay, openCards]
  );

  const submitAction = useCallback(
    async (actionIndex: number) => {
      setError(null);
      try {
        const res = await fetch(`${apiUrl}/api/web/step`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action_index: actionIndex }),
        });
        if (!res.ok) throw new Error(`Step failed: ${res.status}`);
        const resp: EngineResponse = await res.json();
        applyResponse(resp);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Step failed");
        setStatus("error");
      }
    },
    [apiUrl, applyResponse]
  );

  submitRef.current = submitAction;

  const visibleLog = isReplaying ? replayLog : (state?.log ?? []);

  return {
    state,
    engineActions,
    enginePrompt,
    visibleLog,
    isReplaying,
    status,
    error,
    reset,
    submitAction,
  };
}
