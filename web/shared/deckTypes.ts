// ─── Deck Definition (server-driven) ─────────────────────────────────────────
// Deck data comes from GET /api/web/decks which reads .ydk files on the server.
// The .ydk files in assets/decks/ are the single source of truth.

export interface DeckCardEntry {
  code: number;
  name: string;
}

export interface DeckDefinition {
  name: string;
  filename: string;
  main: DeckCardEntry[];
  extra: DeckCardEntry[];
}

export type DeckPayload = { main: number[]; extra: number[] };
