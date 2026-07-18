import { render, waitFor, within } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { DeckSelector } from "./DeckSelector";

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
    expect(
      within(container).queryByText(/No recommender configured/i)
    ).toBeNull();
    expect(
      within(container).queryByText(/Checking for a recommender/i)
    ).not.toBeNull();
  });
});
