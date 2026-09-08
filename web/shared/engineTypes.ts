// ─── Engine Types ────────────────────────────────────────────────────────────
// TypeScript types mirroring the Python web API response from /api/web/*.
// These represent the real ygopro-core engine state, as opposed to the toy
// game types in gameTypes.ts.

/** Card in the player's hand (no position info). */
export interface EngineHandCard {
  code: number;
  name: string;
  type: "monster" | "spell" | "trap" | "unknown";
  attack?: number | null;
  defense?: number | null;
  level?: number;
  link_rating?: number;
}

/** Card occupying a field zone (monster, spell/trap, graveyard, banished). */
export interface EngineFieldCard {
  code: number;
  name: string;
  type: "monster" | "spell" | "trap" | "unknown";
  position: "ATK" | "DEF" | "FACE_DOWN_ATK" | "FACE_DOWN_DEF";
  attack: number | null;
  defense: number | null;
  level: number;
  link_rating?: number;
}

/** The human player's board (full information). */
export interface EnginePlayerBoard {
  hand: EngineHandCard[];
  monsters: (EngineFieldCard | null)[]; // 5 slots
  spells_traps: (EngineFieldCard | null)[]; // 5 slots
  field_zone: EngineFieldCard | null;
  extra_monster_zone: (EngineFieldCard | null)[]; // [seq5, seq6]
  graveyard: EngineFieldCard[];
  banished: EngineFieldCard[];
  extra_deck: EngineHandCard[];
  deck_count: number;
  lp: number;
}

/** The opponent's board. Face-down cards hidden by default; in open-cards mode
 *  the server sends unhidden zones plus ``hand`` and ``extra_deck`` arrays. */
export interface EngineOpponentBoard {
  hand_count: number;
  hand?: EngineHandCard[]; // present in open-cards mode
  monsters: (EngineFieldCard | null)[]; // 5 slots
  spells_traps: (EngineFieldCard | null)[]; // 5 slots
  field_zone: EngineFieldCard | null;
  extra_monster_zone: (EngineFieldCard | null)[]; // [seq5, seq6]
  graveyard: EngineFieldCard[];
  banished: EngineFieldCard[];
  extra_deck_count: number;
  extra_deck?: EngineHandCard[]; // present in open-cards mode
  deck_count: number;
  lp: number;
}

export interface EngineBoard {
  player: EnginePlayerBoard;
  opponent: EngineOpponentBoard;
}

export interface PendingChainEntry {
  chain_link: number;
  card_code: number;
  card_name: string;
  effect_text: string | null;
  controller: number; // 0 = you, 1 = opponent
}

/** Current duel phase and turn info. */
export interface EngineGameState {
  turn: number;
  phase:
    | "draw"
    | "standby"
    | "main1"
    | "battle_start"
    | "battle_step"
    | "damage"
    | "damage_calc"
    | "battle"
    | "main2"
    | "end"
    | "unknown";
  is_my_turn: boolean;
  chain_count: number;
  pending_chain: PendingChainEntry[];
}

/** A single legal action the engine offers. */
export interface EngineAction {
  index: number;
  description: string;
  card_code: number;
  card_name: string;
  controller?: 0 | 1;
  location?: number; // engine location constant (0x02=HAND, 0x04=MZONE, etc.)
  sequence?: number; // slot index
  category:
    | "summon"
    | "special_summon"
    | "reposition"
    | "monster_set"
    | "spell_set"
    | "activate"
    | "to_battle"
    | "to_end"
    | "attack"
    | "to_main2"
    | "yes"
    | "no"
    | "chain"
    | "pass"
    | "select_card"
    | "position"
    | "place"
    | "tribute"
    | "sort"
    | "finish"
    | "option"
    | "unknown";
}

// ─── Prompt metadata ──────────────────────────────────────────────────────────

export type PromptType =
  | "idle_cmd"
  | "battle_cmd"
  | "effect_yn"
  | "yes_no"
  | "option"
  | "select_card"
  | "chain"
  | "place"
  | "position"
  | "tribute"
  | "sort_card"
  | "unknown";

/** A card already picked in a multi-step prompt; minimal data for thumbnail rendering. */
export interface PickedCard {
  code: number;
  location: number;
  controller: number;
  sequence: number;
  param: number;
}

/** Prompt metadata describing the current engine decision context. */
export interface EnginePrompt {
  type: PromptType;
  card_code?: number;
  card_name?: string;
  /** Engine location of the card the prompt targets (effect_yn, position).
   *  Same encoding as EngineAction.location. */
  location?: number;
  min?: number;
  max?: number;
  cancelable?: boolean;
  finishable?: boolean;
  forced?: boolean;
  count?: number;
  /** Cards already picked across multi-step prompts (sort, select_card, tribute, unselect). */
  picked_cards?: PickedCard[];
  selected_count?: number;
  // Tribute-specific fields
  min_release?: number;
  max_cards?: number;
  release_total?: number;
  cards_selected?: number;
  // Yes/no prompt fields (MSG_SELECT_YESNO, MSG_SELECT_EFFECTYN).
  // `desc` is the raw engine u64 (sysstring id or packed card-string ref);
  // `prompt_text` is the server-resolved display string, or null when the
  // engine emitted desc=0 or the resolver couldn't find the string.
  desc?: number;
  prompt_text?: string | null;
}

/** A snapshot of the board + events captured after one engine processing chunk. */
export interface EventFrame {
  events: string[];
  board: EngineBoard;
  game_state: EngineGameState;
}

/**
 * Value-head and policy-head readouts for the current prompt, from the same
 * forward pass that produced the recommendation.
 *
 * Null unless AI Assist is on with a model recommender — random, greedy and
 * ygo-agent recommenders have no value head.
 */
export interface EngineRecommendation {
  /** Matches an `EngineAction.index`. */
  action_index: number;
  /**
   * Raw value-head output for the human player's position: the model's
   * estimate of the discounted return — the terminal win or loss plus any
   * reward shaping it trained with. NOT a win probability. Null for a
   * recommender with no value head, which still picks an index.
   */
  value: number | null;
  /** Policy probabilities, index-aligned with `actions[]`; null with `value`. */
  action_probs: number[] | null;
}

/** Unified response shape from all /api/web/ endpoints. */
export interface EngineResponse {
  board: EngineBoard;
  game_state: EngineGameState;
  actions: EngineAction[];
  prompt: EnginePrompt | null;
  done: boolean;
  reward: number;
  frames: EventFrame[];
  /** What the recommender produced for this prompt, or null when AI-assist is
   *  off / unavailable / terminal. */
  recommendation: EngineRecommendation | null;
}

// ─── Card info (GET /api/web/card/{code}) ────────────────────────────────────

/** Pendulum scales; present only on Pendulum monsters. */
export interface CardScales {
  left: number;
  right: number;
}

/** The printed face of one card, decoded server-side from assets/cards.cdb.
 *  Optional fields arrive as explicit `null`s, never as omitted keys. */
export interface CardInfo {
  code: number;
  name: string;
  desc: string;
  card_type: "monster" | "spell" | "trap" | "unknown";
  /** Printed typeline words, already ordered: ["Dragon","Synchro","Effect"]. */
  typeline: string[];
  attribute: string | null;
  race: string | null;
  /** Level, Rank or Link rating — `level_kind` says which. */
  level: number | null;
  level_kind: "level" | "rank" | "link" | null;
  /** -2 encodes "?" ATK/DEF. */
  attack: number | null;
  defense: number | null;
  scales: CardScales | null;
  /** e.g. ["TOP","BOTTOM_LEFT"]; null for non-Link cards. */
  link_arrows: string[] | null;
}
