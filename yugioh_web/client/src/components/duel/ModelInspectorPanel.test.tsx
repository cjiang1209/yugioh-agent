import { render, cleanup } from "@testing-library/react";
import { describe, it, expect, afterEach } from "vitest";
import { ModelInspectorPanel } from "./ModelInspectorPanel";

afterEach(() => {
  cleanup();
});

describe("ModelInspectorPanel", () => {
  it("shows the signed value to three decimals", () => {
    const { container } = render(<ModelInspectorPanel trace={[0.4123]} />);
    expect(container.textContent).toContain("+0.412");
  });

  it("keeps the sign on a negative value", () => {
    const { container } = render(<ModelInspectorPanel trace={[-0.25]} />);
    expect(container.textContent).toContain("-0.250");
  });

  it("shows the newest sample of the trace", () => {
    const { container } = render(<ModelInspectorPanel trace={[0.1, 0.9]} />);
    expect(container.textContent).toContain("+0.900");
    expect(container.textContent).not.toContain("+0.100");
  });

  it("reports the prompt count", () => {
    const { container } = render(
      <ModelInspectorPanel trace={[0.1, 0.2, 0.3]} />
    );
    expect(container.textContent).toContain("3 prompts");
  });

  it("labels a missing value neutrally, without guessing why", () => {
    // An empty trace means no evaluation has arrived yet this duel, or the
    // recommender has no value head and none ever will. The panel cannot tell
    // those apart, so it names neither. Rendering 0.000 would be worse still:
    // it reads as a real judgement of a level board.
    const { container } = render(<ModelInspectorPanel trace={[]} />);
    expect(container.textContent).toContain("No current evaluation");
    expect(container.textContent).not.toContain("0.000");
    expect(container.textContent).not.toContain("value head");
  });

  it("warns that the value is not a win probability", () => {
    const { getByTitle } = render(<ModelInspectorPanel trace={[0.4]} />);
    expect(getByTitle(/not a win probability/i)).toBeTruthy();
  });
});
