import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Metrics } from "../src/metrics";

describe("Metrics", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("counts tokens generated within the last second", () => {
    const m = new Metrics();
    m.recordToken();
    m.recordToken();
    expect(m.snapshot(0, 0).tokensPerSecond).toBe(2);
  });

  it("drops tokens older than the 1s window", () => {
    const m = new Metrics();
    m.recordToken();
    vi.advanceTimersByTime(1500);
    expect(m.snapshot(0, 0).tokensPerSecond).toBe(0);
  });

  it("keeps lifetime totals independent of the rolling window", () => {
    const m = new Metrics();
    m.recordToken();
    m.recordToken();
    m.recordRequestCompleted();
    vi.advanceTimersByTime(5000); // well outside the 1s window

    const snap = m.snapshot(1, 0);
    expect(snap.tokensPerSecond).toBe(0);
    expect(snap.totalTokensGenerated).toBe(2);
    expect(snap.totalRequestsCompleted).toBe(1);
    expect(snap.activeGenerations).toBe(1);
  });
});
