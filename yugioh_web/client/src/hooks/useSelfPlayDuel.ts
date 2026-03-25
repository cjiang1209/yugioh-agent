// ─── Self-Play Duel Hook ──────────────────────────────────────────────────────
// Runs a complete Yu-Gi-Oh! duel entirely in the browser with no server needed.
// Both sides are controlled by the same player. The active side switches
// automatically when the turn changes, or the player can manually toggle.

import { useCallback, useEffect, useRef, useState } from "react";
import { DuelState, GameAction, PlayerSide, YgoCard } from "../../../shared/gameTypes";

// We import the game engine functions directly on the client side.
// The engine is pure TypeScript with no Node.js dependencies, so it's safe.
import { applyAction, createDuelState } from "../../../server/gameEngine";

export type SelfPlayStatus = "loading" | "dueling" | "ended";

export interface UseSelfPlayDuelReturn {
  state: DuelState | null;
  viewSide: PlayerSide;
  status: SelfPlayStatus;
  sendAction: (action: GameAction) => void;
  switchSide: () => void;
  restart: () => void;
}

// ─── Deck loader (fetches from YGOPRODeck API) ────────────────────────────────

const cardCache = new Map<number, YgoCard>();

async function fetchCard(id: number): Promise<YgoCard | null> {
  if (cardCache.has(id)) return cardCache.get(id)!;
  try {
    const res = await fetch(`https://db.ygoprodeck.com/api/v7/cardinfo.php?id=${id}`);
    if (!res.ok) return null;
    const data = await res.json();
    const card: YgoCard = data.data?.[0] ?? null;
    if (card) cardCache.set(id, card);
    return card;
  } catch {
    return null;
  }
}

async function fetchDeckCards(ids: number[]): Promise<YgoCard[]> {
  const uniqueIds = Array.from(new Set(ids));
  const fetched = await Promise.allSettled(uniqueIds.map(fetchCard));
  const cardMap = new Map<number, YgoCard>();
  fetched.forEach((result, i) => {
    if (result.status === "fulfilled" && result.value) {
      cardMap.set(uniqueIds[i], result.value);
    }
  });
  return ids.map((id) => cardMap.get(id)).filter(Boolean) as YgoCard[];
}

// ─── Deck IDs (Yugi & Kaiba) ─────────────────────────────────────────────────

// Blue-Eyes deck — used for both players
const BLUE_EYES_MAIN = [
  89631139, 89631139, 89631139, // Blue-Eyes White Dragon x3
  38517737, 38517737, 38517737, // The White Stone of Legend x3
  64202399,                     // Blue-Eyes Alternative White Dragon x1
  57043986, 57043986,           // Sage with Eyes of Blue x2
  45467446, 45467446,           // Maiden with Eyes of Blue x2
  71039903, 71039903, 71039903, // Dragon Spirit of White x3
  79814787, 79814787,           // Blue-Eyes Solid Dragon x2
  8240199,  8240199,  8240199,  // Effect Veiler x3
  88241506, 88241506, 88241506, // Ash Blossom & Joyous Spring x3
  48800175, 48800175, 48800175, // Trade-In x3
  38120068, 38120068,           // Cards of Consonance x2
  6853254,  6853254,  6853254,  // Silver's Cry x3
  41620959, 41620959,           // The Melody of Awakening Dragon x2
  24094653, 24094653,           // Polymerization x2
  83764718,                     // Monster Reborn x1
  2295440,                      // Return of the Dragon Lords x1
  5318639,  5318639,            // Mystical Space Typhoon x2
  24224830, 24224830,           // Burst Stream of Destruction x2
];

const BLUE_EYES_EXTRA = [
  23995346,                     // Blue-Eyes Ultimate Dragon x1
  56532353, 56532353,           // Azure-Eyes Silver Dragon x2
  2129638,  2129638,            // Blue-Eyes Twin Burst Dragon x2
  43228023,                     // Blue-Eyes Spirit Dragon x1
  59822133, 59822133,           // Neo Blue-Eyes Ultimate Dragon x2
  40908371, 40908371,           // Blue-Eyes Alternative Ultimate Dragon x2
  89604813,                     // Borreload Savage Dragon x1
];

const BLUE_EYES_IDS = [...BLUE_EYES_MAIN, ...BLUE_EYES_EXTRA];

const YUGI_IDS = BLUE_EYES_IDS;
const KAIBA_IDS = BLUE_EYES_IDS;

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useSelfPlayDuel(): UseSelfPlayDuelReturn {
  const [state, setState] = useState<DuelState | null>(null);
  const [viewSide, setViewSide] = useState<PlayerSide>("player1");
  const [status, setStatus] = useState<SelfPlayStatus>("loading");
  const stateRef = useRef<DuelState | null>(null);

  // Keep ref in sync for use inside callbacks
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const initDuel = useCallback(async () => {
    setStatus("loading");
    setState(null);

    const [yugiDeck, kaibaDeck] = await Promise.all([
      fetchDeckCards(YUGI_IDS),
      fetchDeckCards(KAIBA_IDS),
    ]);

    const initialState = createDuelState(
      "self-play",
      "player1", "Yugi",  yugiDeck,
      "player2", "Kaiba", kaibaDeck  // Both using Blue-Eyes deck
    );

    setState(initialState);
    stateRef.current = initialState;
    setViewSide("player1");
    setStatus("dueling");
  }, []);

  // Auto-start on mount
  useEffect(() => {
    initDuel();
  }, [initDuel]);

  // Auto-switch view side when the active player changes
  useEffect(() => {
    if (state && !state.winner) {
      setViewSide(state.activePlayer);
    }
  }, [state?.activePlayer]);

  const sendAction = useCallback((action: GameAction) => {
    const current = stateRef.current;
    if (!current) return;

    const newState = applyAction(current, action, viewSide);
    setState(newState);
    stateRef.current = newState;

    if (newState.winner) {
      setStatus("ended");
    }
  }, [viewSide]);

  const switchSide = useCallback(() => {
    setViewSide((prev) => (prev === "player1" ? "player2" : "player1"));
  }, []);

  const restart = useCallback(() => {
    initDuel();
  }, [initDuel]);

  return { state, viewSide, status, sendAction, switchSide, restart };
}
