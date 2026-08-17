import { render, within } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { EngineActionPanel } from "./EngineActionPanel";
import type { EngineAction } from "../../../../shared/engineTypes";

const actions: EngineAction[] = [
  {
    index: 3,
    description: "Summon Blue-Eyes",
    card_code: 89631139,
    card_name: "Blue-Eyes White Dragon",
    category: "summon",
  },
  {
    index: 5,
    description: "Set spell",
    card_code: 0,
    card_name: "Face-down",
    category: "spell_set",
  },
];

describe("EngineActionPanel recommended tag", () => {
  it("shows the recommended badge on the matching action only", () => {
    const { container } = render(
      <EngineActionPanel
        actions={actions}
        onAction={() => {}}
        recommendedIndex={3}
      />
    );
    expect(
      within(container).getAllByTitle("Recommended by AI Assist")
    ).toHaveLength(1);
  });

  it("shows no badge when recommendedIndex is null", () => {
    const { container } = render(
      <EngineActionPanel
        actions={actions}
        onAction={() => {}}
        recommendedIndex={null}
      />
    );
    expect(
      within(container).queryAllByTitle("Recommended by AI Assist")
    ).toHaveLength(0);
  });

  it("shows no badge when recommendedIndex matches no action", () => {
    const { container } = render(
      <EngineActionPanel
        actions={actions}
        onAction={() => {}}
        recommendedIndex={99}
      />
    );
    expect(
      within(container).queryAllByTitle("Recommended by AI Assist")
    ).toHaveLength(0);
  });
});

describe("EngineActionPanel probabilities", () => {
  it("shows one percentage per action, aligned by list position", () => {
    // actionProbs is index-aligned with actions[], NOT keyed by action.index:
    // actions[0].index is 3 here, and 0.61 belongs to it.
    const { container } = render(
      <EngineActionPanel
        actions={actions}
        onAction={() => {}}
        actionProbs={[0.61, 0.39]}
      />
    );
    expect(container.textContent).toContain("61%");
    expect(container.textContent).toContain("39%");
  });

  it("renders no percentages when none are supplied", () => {
    const { container } = render(
      <EngineActionPanel actions={actions} onAction={() => {}} />
    );
    expect(container.textContent).not.toMatch(/\d+%/);
  });

  it("omits the percentage for an action the array does not cover", () => {
    // A short array must degrade per-row rather than throwing or showing NaN.
    const { container } = render(
      <EngineActionPanel
        actions={actions}
        onAction={() => {}}
        actionProbs={[0.61]}
      />
    );
    expect(container.textContent).toContain("61%");
    expect(container.textContent).not.toContain("NaN");
  });
});
