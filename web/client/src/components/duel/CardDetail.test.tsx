import { describe, it, expect, vi, afterEach } from "vitest";
import { useLayoutEffect } from "react";
import { cleanup, render, waitFor } from "@testing-library/react";
import { CardDetail } from "./CardDetail";
import { clearInfoCache, getCachedInfo } from "../../lib/cardInfoCache";
import type { GameCard } from "../../../../shared/gameTypes";
import type { CardInfo } from "../../../../shared/engineTypes";

const A_ID = 44508094; // Stardust Dragon
const B_ID = 89631139; // Blue-Eyes White Dragon

function makeCard(over: Partial<GameCard> = {}): GameCard {
  return {
    id: A_ID,
    instanceId: `${A_ID}-mine-hand-0`,
    name: "Stardust Dragon",
    type: "Effect Monster",
    frameType: "effect",
    desc: "",
    card_images: [],
    ...over,
  };
}

function makeInfo(over: Partial<CardInfo> = {}): CardInfo {
  return {
    code: A_ID,
    name: "Stardust Dragon",
    desc: "1 Tuner + 1+ non-Tuner monsters",
    card_type: "monster",
    typeline: ["Dragon", "Synchro", "Effect"],
    attribute: "WIND",
    race: "Dragon",
    level: 8,
    level_kind: "level",
    attack: 2500,
    defense: 2000,
    scales: null,
    link_arrows: null,
    ...over,
  };
}

const okResponse = (info: CardInfo) => ({
  ok: true,
  status: 200,
  json: () => Promise.resolve(info),
});

const notFoundResponse = () => ({
  ok: false,
  status: 404,
  json: () => Promise.resolve({ detail: "Unknown card code" }),
});

function stubFetch(info: CardInfo = makeInfo()) {
  const fetchMock = vi.fn(() => Promise.resolve(okResponse(info)));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

/** A promise plus its resolver, so tests control resolution order explicitly. */
function deferred<T>() {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>(r => {
    resolve = r;
  });
  return { promise, resolve };
}

/** Reports the document's text at every commit, from a layout effect — i.e.
 *  after React has mutated the DOM but before passive effects run. That window
 *  is where a stale-state render is visible; by the time act() returns, any
 *  effect-based correction has already been applied. */
function CommitProbe({ onCommit }: { onCommit: (text: string) => void }) {
  useLayoutEffect(() => {
    onCommit(document.body.textContent ?? "");
  });
  return null;
}

describe("CardDetail", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    clearInfoCache();
  });

  it("renders the printed face: name, attribute + level, typeline, ATK/DEF, text", async () => {
    stubFetch();
    const { findByText, getByText } = render(<CardDetail card={makeCard()} />);

    expect(await findByText("Dragon / Synchro / Effect")).toBeTruthy();
    expect(getByText("Stardust Dragon")).toBeTruthy();
    expect(getByText(/WIND/)).toBeTruthy();
    expect(getByText(/★8/)).toBeTruthy();
    expect(getByText("ATK/2500")).toBeTruthy();
    expect(getByText("DEF/2000")).toBeTruthy();
    expect(getByText(/1 Tuner \+ 1\+ non-Tuner monsters/)).toBeTruthy();
  });

  it("preserves line breaks in a two-section Pendulum description", async () => {
    const desc =
      "[ Pendulum Effect ]\nDuring your Main Phase: …\n" +
      "-".repeat(40) +
      "\n[ Monster Effect ]\nYou can target 1 card …";
    stubFetch(
      makeInfo({
        code: 41546,
        name: "D/D Savant Thomas",
        desc,
        typeline: ["Fiend", "Pendulum", "Effect"],
        race: "Fiend",
        attribute: "DARK",
        scales: { left: 6, right: 6 },
      })
    );

    const { findByText } = render(<CardDetail card={makeCard()} />);
    // Collapsed whitespace would merge the sections into one line, so assert
    // the text node keeps its breaks AND that CSS renders them.
    const paragraph = await findByText(/\[ Pendulum Effect \]/);
    expect(paragraph.textContent).toContain("[ Pendulum Effect ]\n");
    expect(paragraph.textContent).toContain("\n[ Monster Effect ]");
    expect(paragraph.style.whiteSpace).toBe("pre-line");
  });

  it("shows pendulum scales when present", async () => {
    stubFetch(makeInfo({ scales: { left: 6, right: 6 } }));
    const { findByText } = render(<CardDetail card={makeCard()} />);
    expect(await findByText(/◀\s*6/)).toBeTruthy();
    expect(await findByText(/6\s*▶/)).toBeTruthy();
  });

  it("shows link arrows and omits the DEF row for a Link monster", async () => {
    stubFetch(
      makeInfo({
        code: 146746,
        name: "Double Headed Anger Knuckle",
        typeline: ["Machine", "Link", "Effect"],
        race: "Machine",
        attribute: "EARTH",
        level: 2,
        level_kind: "link",
        attack: 1500,
        defense: null,
        link_arrows: ["RIGHT", "BOTTOM"],
      })
    );

    const { findByText, getByText, queryByText, getByTestId } = render(
      <CardDetail card={makeCard()} />
    );
    expect(await findByText(/LINK-2/)).toBeTruthy();
    expect(getByTestId("link-arrow-RIGHT").dataset.lit).toBe("true");
    expect(getByTestId("link-arrow-BOTTOM").dataset.lit).toBe("true");
    expect(getByTestId("link-arrow-TOP").dataset.lit).toBe("false");
    expect(queryByText(/DEF\//)).toBeNull();
    expect(getByText("ATK/1500")).toBeTruthy();
  });

  it("omits attribute, level and ATK/DEF rows for a spell", async () => {
    stubFetch(
      makeInfo({
        code: 483,
        name: "Parallel Teleport",
        card_type: "spell",
        typeline: ["Spell", "Quick-Play"],
        attribute: null,
        race: null,
        level: null,
        level_kind: null,
        attack: null,
        defense: null,
        desc: "Special Summon …",
      })
    );

    const { findByText, queryByText } = render(
      <CardDetail card={makeCard()} />
    );
    expect(await findByText("Spell / Quick-Play")).toBeTruthy();
    expect(queryByText(/ATK\//)).toBeNull();
    expect(queryByText(/DEF\//)).toBeNull();
    expect(queryByText(/★/)).toBeNull();
  });

  it("renders ? for unknown ATK/DEF", async () => {
    stubFetch(makeInfo({ attack: -2, defense: -2 }));
    const { findByText } = render(<CardDetail card={makeCard()} />);
    expect(await findByText("ATK/?")).toBeTruthy();
    expect(await findByText("DEF/?")).toBeTruthy();
  });

  it("renders the board name without stats when no face is available, and caches the 404", async () => {
    // A 404 is the cheapest way to reach the no-face render path; it is not a
    // realistic board state. Board names come from the same cards.cdb as the
    // card face (board_state.py -> get_card_name), so a passcode the endpoint
    // cannot find would already read "Unknown(code)" on the board. What this
    // path really covers is the window before a fetch resolves.
    const fetchMock = vi.fn(() => Promise.resolve(notFoundResponse()));
    vi.stubGlobal("fetch", fetchMock);

    // A name that appears nowhere else — not in makeCard's default, not in any
    // mocked CardInfo — so this assertion can only pass if the rendered name
    // came from the board card. With the default name it matched makeInfo()'s
    // name too, and so could not distinguish fallback from a rendered face.
    const { getByText, queryByText } = render(
      <CardDetail card={makeCard({ name: "Name From The Board" })} />
    );
    await waitFor(() => expect(getCachedInfo(A_ID)).toBeNull());
    expect(getByText("Name From The Board")).toBeTruthy();
    expect(queryByText(/ATK\//)).toBeNull();
    cleanup();

    render(<CardDetail card={makeCard()} />);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("re-fetches after a transient failure", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new Error("network down"))
      .mockResolvedValue(okResponse(makeInfo()));
    vi.stubGlobal("fetch", fetchMock);

    render(<CardDetail card={makeCard()} />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    expect(getCachedInfo(A_ID)).toBeUndefined(); // nothing poisoned
    cleanup();

    const retry = render(<CardDetail card={makeCard()} />);
    expect(await retry.findByText("Dragon / Synchro / Effect")).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("never fetches for a face-down card (id 0)", () => {
    const fetchMock = stubFetch();
    render(<CardDetail card={makeCard({ id: 0, name: "Card" })} />);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("drops the previous card's face when switching", async () => {
    stubFetch();
    const { rerender, findByText, queryByText } = render(
      <CardDetail card={makeCard()} />
    );
    expect(await findByText("Dragon / Synchro / Effect")).toBeTruthy();

    rerender(<CardDetail card={makeCard({ id: B_ID, name: "Blue-Eyes" })} />);
    expect(queryByText("Dragon / Synchro / Effect")).toBeNull();
  });

  it("never commits the previous card's face after a switch, not even for one frame", async () => {
    // The test above cannot catch this: `rerender` wraps in act(), which
    // flushes effects before the assertion runs, so a component that renders
    // from stale state has already corrected itself by then. React commits a
    // render to the DOM *before* flushing passive effects, so the only way to
    // observe the bad frame is a layout effect, which runs in between.
    const infoFor = (id: number) =>
      id === A_ID
        ? makeInfo()
        : makeInfo({
            code: B_ID,
            name: "Blue-Eyes White Dragon",
            typeline: ["Dragon", "Normal"],
          });
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) =>
        Promise.resolve(
          okResponse(infoFor(url.includes(`${A_ID}`) ? A_ID : B_ID))
        )
      )
    );

    // Prime both cards, so the effect after the switch has nothing to correct
    // and the first committed render is the only thing under test.
    const primed = render(<CardDetail card={makeCard()} />);
    await primed.findByText("Dragon / Synchro / Effect");
    cleanup();
    const primedB = render(
      <CardDetail card={makeCard({ id: B_ID, name: "Blue-Eyes" })} />
    );
    await primedB.findByText("Dragon / Normal");
    cleanup();

    const commits: string[] = [];
    const probe = (text: string) => commits.push(text);
    const { rerender } = render(
      <>
        <CardDetail card={makeCard()} />
        <CommitProbe onCommit={probe} />
      </>
    );
    commits.length = 0; // ignore the initial mount

    rerender(
      <>
        <CardDetail card={makeCard({ id: B_ID, name: "Blue-Eyes" })} />
        <CommitProbe onCommit={probe} />
      </>
    );

    expect(commits.length).toBeGreaterThan(0);
    for (const text of commits) {
      expect(text).not.toContain("Dragon / Synchro / Effect");
      expect(text).not.toContain("Stardust Dragon");
    }
  });

  it("ignores an earlier request that resolves after a card switch", async () => {
    const a = deferred<ReturnType<typeof okResponse>>();
    const b = deferred<ReturnType<typeof okResponse>>();
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => (url.includes(`${A_ID}`) ? a.promise : b.promise))
    );

    const { rerender, findByText, queryByText } = render(
      <CardDetail card={makeCard()} />
    );
    rerender(<CardDetail card={makeCard({ id: B_ID, name: "Blue-Eyes" })} />);

    // B resolves first, then A — the stale response must not win.
    b.resolve(
      okResponse(
        makeInfo({
          code: B_ID,
          name: "Blue-Eyes White Dragon",
          typeline: ["Dragon", "Normal"],
        })
      )
    );
    expect(await findByText("Dragon / Normal")).toBeTruthy();

    a.resolve(okResponse(makeInfo()));
    // Wait on a signal that A's promise chain actually ran: polling for the
    // absence of A's typeline would pass before the chain executes and prove
    // nothing. The cache write happens in the same `.then` as the (suppressed)
    // state update, so it is the reliable signal.
    await waitFor(() => expect(getCachedInfo(A_ID)).not.toBeUndefined());
    expect(queryByText("Dragon / Synchro / Effect")).toBeNull();
    expect(queryByText("Dragon / Normal")).toBeTruthy();
  });
});
