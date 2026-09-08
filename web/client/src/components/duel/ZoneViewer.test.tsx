import { describe, it, expect, afterEach } from "vitest";
import { cleanup, render } from "@testing-library/react";
import { ZoneViewer } from "./ZoneViewer";
import { PANEL_WIDTH } from "../../lib/duelLayout";

afterEach(cleanup);

describe("ZoneViewer", () => {
  it("reserves both side panels via the backdrop's left/right inset", () => {
    const { container } = render(
      <ZoneViewer
        graveyard={[]}
        banished={[]}
        playerName="You"
        onClose={() => {}}
      />
    );

    // ZoneViewer's root element is the fixed, full-height backdrop that
    // insets left/right to leave both side panels visible.
    const backdrop = container.firstChild as HTMLElement;

    expect(backdrop.style.left).toBe(`${PANEL_WIDTH}px`);
    expect(backdrop.style.right).toBe(`${PANEL_WIDTH}px`);
  });
});
