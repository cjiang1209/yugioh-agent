import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { ChainWidget } from "./ChainWidget";
import type { PendingChainEntry } from "../../../../shared/engineTypes";

const entry = (over: Partial<PendingChainEntry> = {}): PendingChainEntry => ({
  chain_link: 1,
  card_code: 41420027,
  card_name: "Solemn Judgment",
  effect_text: "Negate the summon",
  controller: 1,
  ...over,
});

describe("ChainWidget", () => {
  it("renders nothing when empty", () => {
    const { container } = render(<ChainWidget entries={[]} />);
    expect(container.firstChild).toBeNull();
  });
  it("renders one row per link with name, effect, YOU/OPP", () => {
    const { getAllByText } = render(
      <ChainWidget
        entries={[entry(), entry({ chain_link: 2, controller: 0 })]}
      />
    );
    expect(getAllByText(/Solemn Judgment/).length).toBeGreaterThan(0);
    expect(getAllByText(/Negate the summon/).length).toBeGreaterThan(0);
    expect(getAllByText(/OPP/).length).toBeGreaterThan(0);
    expect(getAllByText(/YOU/).length).toBeGreaterThan(0);
  });
});
