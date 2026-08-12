import { MetricsSnapshot } from "./types";

// Tracks a rolling window of token-generation timestamps so
// tokensPerSecond reflects "the last second", not an all-time average
// that barely moves once the server has been up for a while.
export class Metrics {
  private tokenTimestamps: number[] = [];
  private totalTokensGenerated = 0;
  private totalRequestsCompleted = 0;

  recordToken(): void {
    this.totalTokensGenerated++;
    this.tokenTimestamps.push(Date.now());
  }

  recordRequestCompleted(): void {
    this.totalRequestsCompleted++;
  }

  private tokensPerSecond(): number {
    const cutoff = Date.now() - 1000;
    // Drop anything older than 1s off the front. Timestamps only ever
    // get pushed onto the back in order, so this stays cheap even under
    // sustained load -- no need to scan the whole array each time.
    while (this.tokenTimestamps.length && this.tokenTimestamps[0] < cutoff) {
      this.tokenTimestamps.shift();
    }
    return this.tokenTimestamps.length;
  }

  snapshot(activeGenerations: number, queueDepth: number): MetricsSnapshot {
    return {
      activeGenerations,
      queueDepth,
      totalRequestsCompleted: this.totalRequestsCompleted,
      totalTokensGenerated: this.totalTokensGenerated,
      tokensPerSecond: this.tokensPerSecond(),
    };
  }
}
