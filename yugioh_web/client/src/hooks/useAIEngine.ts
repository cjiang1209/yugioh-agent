// ─── AI Engine Hook ─────────────────────────────────────────────────────────
// Connects to the Python FastAPI backend (/api/web/*) and maps responses
// to the existing DuelState shape consumed by DuelBoard.

import { useCallback, useRef, useState } from "react";
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
  EngineFieldCard,
  EngineHandCard,
  EnginePrompt,
  EngineResponse,
} from "../../../shared/engineTypes";

export type AIEngineStatus = "idle" | "loading" | "dueling" | "ended" | "error";

export interface UseAIEngineReturn {
  state: DuelState | null;
  engineActions: EngineAction[];
  enginePrompt: EnginePrompt | null;
  eventLog: string[];
  status: AIEngineStatus;
  error: string | null;
  reset: (seed?: number, deck0?: DeckPayload, deck1?: DeckPayload) => Promise<void>;
  submitAction: (actionIndex: number) => Promise<void>;
}

// ─── Mapping helpers ─────────────────────────────────────────────────────────

const CARD_IMAGE_BASE = "https://images.ygoprodeck.com/images/cards_small";

let _instanceCounter = 0;

function engineCardToGameCard(
  card: EngineHandCard | EngineFieldCard,
  zone: string,
  seq: number,
): GameCard {
  const id = card.code || 0;
  const instanceId = `e-${id}-${zone}-${seq}-${_instanceCounter++}`;
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
      ? [{ id, image_url: `${CARD_IMAGE_BASE}/${id}.jpg`, image_url_small: `${CARD_IMAGE_BASE}/${id}.jpg` }]
      : [],
  };
}

function mapCardType(t: string): string {
  switch (t) {
    case "monster": return "Effect Monster";
    case "spell": return "Spell Card";
    case "trap": return "Trap Card";
    default: return "Effect Monster";
  }
}

function mapPosition(pos: string): CardPosition {
  switch (pos) {
    case "ATK": return "ATK";
    case "DEF": return "DEF";
    case "FACE_DOWN_DEF": return "FACE_DOWN_DEF";
    case "FACE_DOWN_ATK": return "FACE_DOWN_ATK";
    default: return "ATK";
  }
}

function engineFieldCardToFieldCard(
  card: EngineFieldCard,
  zone: string,
  seq: number,
): FieldCard {
  const position = mapPosition(card.position);
  return {
    card: engineCardToGameCard(card, zone, seq),
    position,
    faceDown: position === "FACE_DOWN_DEF" || position === "FACE_DOWN_ATK",
  };
}

function mapPhase(phase: string): Phase {
  switch (phase) {
    case "draw": return "DRAW";
    case "standby": return "STANDBY";
    case "main1": return "MAIN1";
    case "battle_start":
    case "battle_step":
    case "battle":
    case "damage":
    case "damage_calc":
      return "BATTLE";
    case "main2": return "MAIN2";
    case "end": return "END";
    default: return "MAIN1";
  }
}

function makeFaceDownHandCard(index: number): GameCard {
  return {
    id: 0,
    instanceId: `e-opp-hand-${index}-${_instanceCounter++}`,
    name: "Card",
    type: "Effect Monster",
    frameType: "effect",
    desc: "",
    card_images: [],
  };
}

function engineResponseToDuelState(resp: EngineResponse, log: string[]): DuelState {
  const { board, game_state } = resp;

  // Player (human, always player1 / bottom)
  const playerMonsters: (FieldCard | null)[] = (board.player.monsters ?? []).map(
    (m, i) => m ? engineFieldCardToFieldCard(m, "mzone", i) : null,
  );
  const playerST: (FieldCard | null)[] = (board.player.spells_traps ?? []).map(
    (m, i) => m ? engineFieldCardToFieldCard(m, "szone", i) : null,
  );

  const player: PlayerState = {
    id: "player1",
    name: "You",
    lifePoints: board.player.lp,
    hand: (board.player.hand ?? []).map((c, i) => engineCardToGameCard(c, "hand", i)),
    deck: Array.from({ length: board.player.deck_count }, (_, i) => makeFaceDownHandCard(i)),
    graveyard: (board.player.graveyard ?? []).map((c, i) => engineCardToGameCard(c, "grave", i)),
    banished: (board.player.banished ?? []).map((c, i) => engineCardToGameCard(c, "banished", i)),
    extraDeck: (board.player.extra_deck ?? []).map((c, i) => engineCardToGameCard(c, "extra", i)),
    monsterZones: playerMonsters,
    spellTrapZones: playerST,
    fieldZone: board.player.field_zone
      ? engineFieldCardToFieldCard(board.player.field_zone, "field", 0)
      : null,
    extraMonsterZone: null,
    hasNormalSummoned: false,
    hasDrawn: false,
  };

  // Opponent (always player2 / top)
  const oppMonsters: (FieldCard | null)[] = (board.opponent.monsters ?? []).map(
    (m, i) => m ? engineFieldCardToFieldCard(m, "opp-mzone", i) : null,
  );
  const oppST: (FieldCard | null)[] = (board.opponent.spells_traps ?? []).map(
    (m, i) => m ? engineFieldCardToFieldCard(m, "opp-szone", i) : null,
  );

  const opponent: PlayerState = {
    id: "player2",
    name: "Opponent",
    lifePoints: board.opponent.lp,
    hand: Array.from({ length: board.opponent.hand_count }, (_, i) => makeFaceDownHandCard(i)),
    deck: Array.from({ length: board.opponent.deck_count }, (_, i) => makeFaceDownHandCard(i)),
    graveyard: (board.opponent.graveyard ?? []).map((c, i) => engineCardToGameCard(c, "opp-grave", i)),
    banished: (board.opponent.banished ?? []).map((c, i) => engineCardToGameCard(c, "opp-banished", i)),
    extraDeck: Array.from({ length: board.opponent.extra_deck_count }, (_, i) => makeFaceDownHandCard(i)),
    monsterZones: oppMonsters,
    spellTrapZones: oppST,
    fieldZone: board.opponent.field_zone
      ? engineFieldCardToFieldCard(board.opponent.field_zone, "opp-field", 0)
      : null,
    extraMonsterZone: null,
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
    winner: resp.done
      ? resp.reward > 0
        ? "player1"
        : resp.reward < 0
          ? "player2"
          : null
      : null,
    battleStep: null,
    log,
  };
}

// ─── Hook ────────────────────────────────────────────────────────────────────

export function useAIEngine(apiUrl: string = "http://localhost:8000"): UseAIEngineReturn {
  const [state, setState] = useState<DuelState | null>(null);
  const [engineActions, setEngineActions] = useState<EngineAction[]>([]);
  const [enginePrompt, setEnginePrompt] = useState<EnginePrompt | null>(null);
  const [eventLog, setEventLog] = useState<string[]>([]);
  const [status, setStatus] = useState<AIEngineStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<string[]>([]);

  // Auto-pass categories that should be submitted without prompting
  const AUTO_PASS_CATEGORIES = new Set(["pass", "no"]);

  const submitRef = useRef<(index: number) => Promise<void>>(undefined);

  const applyResponse = useCallback((resp: EngineResponse) => {
    // Append new events to the running log
    const newLog = [...logRef.current, ...resp.event_log];
    logRef.current = newLog;
    setEventLog(newLog);
    setState(engineResponseToDuelState(resp, newLog));
    setStatus(resp.done ? "ended" : "dueling");
    setError(null);
    setEnginePrompt(resp.prompt ?? null);

    // Auto-pass: if the only action is pass/no-chain, submit it automatically
    if (
      !resp.done &&
      resp.actions.length === 1 &&
      AUTO_PASS_CATEGORIES.has(resp.actions[0].category)
    ) {
      setEngineActions([]);  // hide from UI during auto-pass
      setEnginePrompt(null); // hide prompt during auto-pass
      setTimeout(() => submitRef.current?.(resp.actions[0].index), 0);
    } else {
      setEngineActions(resp.actions);
    }
  }, []);

  const reset = useCallback(async (seed?: number, deck0?: DeckPayload, deck1?: DeckPayload) => {
    setStatus("loading");
    setError(null);
    logRef.current = [];
    try {
      const body: Record<string, unknown> = {};
      if (seed !== undefined) body.seed = seed;
      if (deck0 !== undefined) body.deck0 = deck0;
      if (deck1 !== undefined) body.deck1 = deck1;
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
  }, [apiUrl, applyResponse]);

  const submitAction = useCallback(async (actionIndex: number) => {
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
  }, [apiUrl, applyResponse]);

  submitRef.current = submitAction;

  return { state, engineActions, enginePrompt, eventLog, status, error, reset, submitAction };
}
