import { render, cleanup } from "@testing-library/react";
import { describe, it, expect, afterEach } from "vitest";
import { Sparkline } from "./Sparkline";

afterEach(() => {
  cleanup();
});

describe("Sparkline", () => {
  it("draws one polyline point per value, negative below positive", () => {
    const { container } = render(<Sparkline values={[0.1, -0.2, 0.3]} />);
    const points = container.querySelector("polyline")?.getAttribute("points");
    const coords = points?.trim().split(/\s+/) ?? [];
    expect(coords).toHaveLength(3);

    // SVG y grows downward, so the negative value's point must sit below
    // (greater y than) the positive value's point.
    const yOf = (coord: string) => Number(coord.split(",")[1]);
    expect(yOf(coords[1])).toBeGreaterThan(yOf(coords[0]));
  });

  it("scales to the largest magnitude and prints it, so a small wobble is not misread as a swing", () => {
    // The largest magnitude here belongs to the negative sample.
    const { container } = render(<Sparkline values={[0.04, -0.05]} />);
    expect(container.textContent).toContain("±0.05");
  });

  it("projects an all-zero series onto the baseline rather than dividing by zero", () => {
    const { container } = render(<Sparkline values={[0, 0]} />);
    const points = container.querySelector("polyline")?.getAttribute("points");
    expect(points).not.toContain("NaN");
    expect(container.textContent).toContain("±0.01");
  });

  it("always draws the zero baseline", () => {
    const { container } = render(<Sparkline values={[0.5, 0.6]} />);
    expect(container.querySelector("line")).not.toBeNull();
  });

  it("renders no polyline for an empty series, and still prints a scale", () => {
    const { container } = render(<Sparkline values={[]} />);
    expect(container.querySelector("polyline")).toBeNull();
    expect(container.textContent).toContain("±0.01");
  });
});
