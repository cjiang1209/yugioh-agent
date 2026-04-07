// ─── Deck Definition (server-driven) ─────────────────────────────────────────
// Deck data comes from GET /api/web/decks which reads .ydk files on the server.
// The .ydk files in assets/decks/ are the single source of truth.

export interface DeckDefinition {
  name: string;
  filename: string;
  main: number[];
  extra: number[];
}

export type DeckPayload = Pick<DeckDefinition, "main" | "extra">;
