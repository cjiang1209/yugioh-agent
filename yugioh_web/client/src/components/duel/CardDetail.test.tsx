import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { cleanup, render, waitFor } from "@testing-library/react";
import { CardDetail } from "./CardDetail";
import { clearDescCache, getCachedDesc } from "../../lib/cardDescCache";
import type { GameCard } from "../../../../shared/gameTypes";

const A_ID = 46986414; // Dark Magician
const B_ID = 89631139; // Blue-Eyes White Dragon

function makeCard(over: Partial<GameCard> = {}): GameCard {
  return {
    id: A_ID,
    instanceId: `${A_ID}-mine-hand-0`,
    name: "Dark Magician",
    type: "Effect Monster",
    frameType: "effect",
    desc: "",
    atk: 2500,
    def: 2100,
    level: 7,
    race: "Spellcaster",
    attribute: "DARK",
    card_images: [],
    ...over,
  };
}

/** A promise plus its resolver, so tests control resolution order explicitly. */
function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>(r => {
    resolve = r;
  });
  return { promise, resolve };
}

const descResponse = (desc: string) => ({
  ok: true,
  json: () => Promise.resolve({ data: [{ desc }] }),
});

/** What YGOProDeck answers with when it rate-limits: a non-ok error body. */
const rateLimitedResponse = () => ({
  ok: false,
  status: 429,
  json: () => Promise.resolve({ error: "Rate limit exceeded" }),
});

describe("CardDetail", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.resolve(descResponse("fetched text")))
    );
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    clearDescCache();
  });

  it("renders name, type line and description from the card", () => {
    const { getByText } = render(
      <CardDetail card={makeCard({ desc: "Ultimate wizard." })} />
    );
    expect(getByText("Dark Magician")).toBeTruthy();
    expect(getByText(/Effect Monster · ★7 · DARK/)).toBeTruthy();
    expect(getByText("Ultimate wizard.")).toBeTruthy();
  });

  it("shows ATK/DEF for a monster and omits them for a spell", () => {
    const monster = render(<CardDetail card={makeCard({ desc: "x" })} />);
    expect(monster.getByText("ATK/2500")).toBeTruthy();
    expect(monster.getByText("DEF/2100")).toBeTruthy();
    cleanup();

    const spell = render(
      <CardDetail
        card={makeCard({
          id: 24094653,
          name: "Polymerization",
          type: "Spell Card",
          desc: "Fusion Summon.",
          atk: undefined,
          def: undefined,
          level: undefined,
          attribute: undefined,
        })}
      />
    );
    expect(spell.queryByText(/ATK\//)).toBeNull();
    expect(spell.queryByText(/DEF\//)).toBeNull();
  });

  it("fetches the description when the card has none", async () => {
    const { findByText } = render(<CardDetail card={makeCard()} />);
    expect(await findByText("fetched text")).toBeTruthy();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("reuses the cache on a second mount of the same card", async () => {
    const { findByText } = render(<CardDetail card={makeCard()} />);
    await findByText("fetched text");
    cleanup();

    const second = render(<CardDetail card={makeCard()} />);
    expect(second.getByText("fetched text")).toBeTruthy();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it("retries after a rate-limited response instead of caching a blank", async () => {
    // The cache is consulted with `!== undefined`, so caching "" from an error
    // body would suppress every later attempt and leave the card blank for the
    // rest of the session.
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(rateLimitedResponse())
      .mockResolvedValue(descResponse("recovered text"));
    vi.stubGlobal("fetch", fetchMock);

    render(<CardDetail card={makeCard()} />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(getCachedDesc(A_ID)).toBeUndefined(); // nothing poisoned
    cleanup();

    const retry = render(<CardDetail card={makeCard()} />);
    expect(await retry.findByText("recovered text")).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("never fetches for a face-down card (id 0)", () => {
    render(<CardDetail card={makeCard({ id: 0, name: "Card" })} />);
    expect(fetch).not.toHaveBeenCalled();
  });

  it("drops the previous card's description when switching", async () => {
    const { rerender, findByText, queryByText } = render(
      <CardDetail card={makeCard({ desc: "A text" })} />
    );
    expect(await findByText("A text")).toBeTruthy();

    rerender(<CardDetail card={makeCard({ id: B_ID, name: "Blue-Eyes" })} />);
    expect(queryByText("A text")).toBeNull();
  });

  it("ignores an earlier request that resolves after a card switch", async () => {
    const a = deferred<{ ok: boolean; json: () => Promise<unknown> }>();
    const b = deferred<{ ok: boolean; json: () => Promise<unknown> }>();
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => (url.includes(`${A_ID}`) ? a.promise : b.promise))
    );

    const { rerender, findByText, queryByText } = render(
      <CardDetail card={makeCard()} />
    );
    rerender(<CardDetail card={makeCard({ id: B_ID, name: "Blue-Eyes" })} />);

    // B resolves first, then A — the stale response must not win.
    b.resolve(descResponse("B text"));
    expect(await findByText("B text")).toBeTruthy();

    a.resolve(descResponse("A text"));
    // Wait on a signal that A's promise chain actually ran. Polling for the
    // absence of "A text" would pass immediately, before the chain executes,
    // and the test would prove nothing. The cache write happens in the same
    // `.then` as the (suppressed) re-render, so it is the reliable signal.
    await waitFor(() => expect(getCachedDesc(A_ID)).toBe("A text"));
    expect(queryByText("A text")).toBeNull();
    expect(queryByText("B text")).toBeTruthy();
  });
});
