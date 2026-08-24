import { render, within } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { EngineActionPanel } from "./EngineActionPanel";
import type { EngineAction } from "../../../../shared/engineTypes";

// Indices differ from positions, so anything keyed by position instead of by
// `action.index` shows the wrong action's data. Both the badge and the
// percentages key on `action.index`.
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
  it("shows each action's percentage, keyed by its index", () => {
    const { container } = render(
      <EngineActionPanel
        actions={actions}
        onAction={() => {}}
        actionProbs={[0.2, 0.3, 0.05, 0.61, 0.06, 0.39]}
      />
    );
    expect(container.textContent).toContain("61%");
    expect(container.textContent).toContain("39%");
    // The values sitting at positions 0 and 1 belong to actions not shown.
    expect(container.textContent).not.toContain("20%");
    expect(container.textContent).not.toContain("30%");
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
        actionProbs={[0.2, 0.3, 0.05, 0.61]}
      />
    );
    expect(container.textContent).toContain("61%");
    expect(container.textContent).not.toContain("NaN");
    expect(container.textContent?.match(/\d+%/g)).toHaveLength(1);
  });
});

describe("EngineActionPanel probability placement", () => {
  it("pins the readout out of flow, in the star's container", () => {
    // Out of flow is what stops showing or hiding it from moving anything
    // around it; sharing the star's container is what puts it in the same
    // corner here as on every other panel.
    const { container } = render(
      <EngineActionPanel
        actions={actions}
        onAction={() => {}}
        recommendedIndex={3}
        actionProbs={[0.2, 0.3, 0.05, 0.61, 0.06, 0.39]}
      />
    );
    const readout = container.querySelector<HTMLElement>(
      '[data-testid="action-probability"]'
    )!;
    const star = container.querySelector<HTMLElement>(
      '[title="Recommended by AI Assist"]'
    )!;

    expect(readout.style.position).toBe("absolute");
    expect(readout.parentElement).toBe(star.parentElement);
  });
});
