import { fireEvent, render, waitFor, within } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { DeckSelector } from "./DeckSelector";

/** Opens a deck Select (by its position among the two slots, 0 = my deck,
 *  1 = opponent deck) and picks the deck by name. Radix Select renders its
 *  listbox in a portal, so the option is queried against `document.body`
 *  rather than the render container. */
async function pickDeck(
  container: HTMLElement,
  slotIndex: number,
  deckName: string
) {
  const triggers = container.querySelectorAll('[data-slot="select-trigger"]');
  fireEvent.click(triggers[slotIndex]);
  const option = await within(document.body).findByText(deckName);
  fireEvent.click(option);
}

const DECKS = [
  {
    name: "Blue Eyes",
    filename: "blue_eyes.ydk",
    main: [{ code: 89631139, name: "Blue-Eyes White Dragon" }],
    extra: [],
  },
];

function mockFetch(recommendAvailable: boolean) {
  return vi.fn((url: string) => {
    if (url.endsWith("/api/web/decks")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(DECKS) });
    }
    if (url.endsWith("/api/web/config")) {
      return Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({ recommend_available: recommendAvailable }),
      });
    }
    return Promise.reject(new Error(`unexpected url ${url}`));
  });
}

describe("DeckSelector AI-assist toggle", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", mockFetch(true));
    // jsdom doesn't implement these; Radix Select calls them when its
    // listbox opens and an item is selected.
    Element.prototype.scrollIntoView = vi.fn();
    Element.prototype.hasPointerCapture = vi.fn().mockReturnValue(false);
    Element.prototype.setPointerCapture = vi.fn();
    Element.prototype.releasePointerCapture = vi.fn();
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows an enabled AI-assist toggle when recommend is available", async () => {
    const { container } = render(<DeckSelector onDeckSelected={() => {}} />);
    const toggle = await within(container).findByRole("button", {
      name: /AI ASSIST/i,
    });
    expect((toggle as HTMLButtonElement).disabled).toBe(false);
  });

  it("disables the toggle when recommend is unavailable", async () => {
    vi.stubGlobal("fetch", mockFetch(false));
    const { container } = render(<DeckSelector onDeckSelected={() => {}} />);
    const toggle = await within(container).findByRole("button", {
      name: /AI ASSIST/i,
    });
    await waitFor(() =>
      expect((toggle as HTMLButtonElement).disabled).toBe(true)
    );
  });

  it("shows a 'checking' state, not 'unavailable', while config is still loading", async () => {
    // Decks resolve immediately; config never resolves within the test, so the
    // toggle renders in its pre-config state.
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) => {
        if (url.endsWith("/api/web/decks")) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(DECKS),
          });
        }
        if (url.endsWith("/api/web/config")) {
          return new Promise(() => {}); // pending forever
        }
        return Promise.reject(new Error(`unexpected url ${url}`));
      })
    );
    const { container } = render(<DeckSelector onDeckSelected={() => {}} />);
    const toggle = await within(container).findByRole("button", {
      name: /AI ASSIST/i,
    });
    // Disabled while checking, but must NOT claim no recommender is configured.
    expect((toggle as HTMLButtonElement).disabled).toBe(true);
    expect(toggle.textContent).toContain("OFF");
    expect(
      within(container).queryByText(/No recommender configured/i)
    ).toBeNull();
    expect(
      within(container).queryByText(/Checking for a recommender/i)
    ).not.toBeNull();
  });

  it("turns AI assist on once a recommender is known to be available", async () => {
    const { container } = render(<DeckSelector onDeckSelected={() => {}} />);
    const toggle = await within(container).findByRole("button", {
      name: /AI ASSIST/i,
    });
    await waitFor(() => expect(toggle.textContent).toContain("ON"));
  });

  it("leaves AI assist off when no recommender is available", async () => {
    vi.stubGlobal("fetch", mockFetch(false));
    const { container } = render(<DeckSelector onDeckSelected={() => {}} />);
    const toggle = await within(container).findByRole("button", {
      name: /AI ASSIST/i,
    });
    await waitFor(() =>
      expect((toggle as HTMLButtonElement).disabled).toBe(true)
    );
    expect(toggle.textContent).toContain("OFF");
  });

  it("never sends recommend=true to a server with no recommender", async () => {
    // The contract is the confirm callback's `recommend` argument rather than
    // the toggle's rendered text: it must stay false through a real deck pick.
    vi.stubGlobal("fetch", mockFetch(false));
    const onDeckSelected = vi.fn();
    const { container } = render(
      <DeckSelector onDeckSelected={onDeckSelected} />
    );
    await within(container).findByText("MY DECK");
    await pickDeck(container, 0, "Blue Eyes");
    await pickDeck(container, 1, "Blue Eyes");
    const confirm = within(container).getByRole("button", {
      name: /ENTER THE DUEL/i,
    });
    fireEvent.click(confirm);

    expect(onDeckSelected).toHaveBeenCalledTimes(1);
    // onDeckSelected(myDeck, oppDeck, openCards, turnOrder, agentPlayer,
    // animateCoinFlip, recommend) — recommend is the 7th argument.
    expect(onDeckSelected.mock.calls[0][6]).toBe(false);
  });
});
