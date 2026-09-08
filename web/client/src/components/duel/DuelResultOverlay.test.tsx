import { describe, it, expect, vi, afterEach } from "vitest";
import { cleanup, fireEvent, render } from "@testing-library/react";
import { DuelResultOverlay } from "./DuelResultOverlay";

describe("DuelResultOverlay", () => {
  // No `globals: true` in vitest.config.ts, so RTL's auto-cleanup is not
  // registered — unmount between renders ourselves.
  afterEach(cleanup);

  it.each([
    ["win", "VICTORY"],
    ["loss", "DEFEAT"],
    ["draw", "DRAW"],
  ] as const)("renders the %s title", (outcome, title) => {
    const { getByText } = render(<DuelResultOverlay outcome={outcome} />);
    expect(getByText(title)).toBeTruthy();
  });

  it("shows the closing log line", () => {
    const { getByText } = render(
      <DuelResultOverlay outcome="win" lastLogLine="Opponent's LP reached 0" />
    );
    expect(getByText("Opponent's LP reached 0")).toBeTruthy();
  });

  it("fires each handler on click", () => {
    const onRestart = vi.fn();
    const onChangeDecks = vi.fn();
    const onDismiss = vi.fn();
    const { getByText } = render(
      <DuelResultOverlay
        outcome="loss"
        onRestart={onRestart}
        onChangeDecks={onChangeDecks}
        onDismiss={onDismiss}
      />
    );
    fireEvent.click(getByText("DUEL AGAIN"));
    fireEvent.click(getByText("CHANGE DECKS"));
    fireEvent.click(getByText("VIEW BOARD"));
    expect(onRestart).toHaveBeenCalledOnce();
    expect(onChangeDecks).toHaveBeenCalledOnce();
    expect(onDismiss).toHaveBeenCalledOnce();
  });

  it("omits buttons without a handler", () => {
    const { queryByText } = render(<DuelResultOverlay outcome="draw" />);
    expect(queryByText("DUEL AGAIN")).toBeNull();
    expect(queryByText("CHANGE DECKS")).toBeNull();
    expect(queryByText("VIEW BOARD")).toBeNull();
  });
});
