import { describe, it, expect, vi, afterEach } from "vitest";
import { resolveTurnOrder } from "./turnOrder";

describe("resolveTurnOrder", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns agentPlayer=0 and no animation for 'first'", () => {
    expect(resolveTurnOrder("first")).toEqual({ agentPlayer: 0, animateCoinFlip: false });
  });

  it("returns agentPlayer=1 and no animation for 'second'", () => {
    expect(resolveTurnOrder("second")).toEqual({ agentPlayer: 1, animateCoinFlip: false });
  });

  it("returns agentPlayer=0 and animation for 'random' when Math.random() < 0.5", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.4);
    expect(resolveTurnOrder("random")).toEqual({ agentPlayer: 0, animateCoinFlip: true });
  });

  it("returns agentPlayer=1 and animation for 'random' when Math.random() >= 0.5", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.6);
    expect(resolveTurnOrder("random")).toEqual({ agentPlayer: 1, animateCoinFlip: true });
  });
});
