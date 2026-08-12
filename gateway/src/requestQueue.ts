import { QueuedRequest } from "./types";

// Admission control for whole requests, sitting in front of the engine.
// This is a coarser-grained cousin of the continuous-batching scheduler
// on the Python side: that one admits individual *sequences* into a
// token-level batch as KV-cache blocks free up; this one admits whole
// *requests* into a fixed number of concurrent generation slots, and
// queues or rejects the rest. Different layer, same underlying idea --
// don't let more work in than the thing underneath can actually handle.
export type AdmissionResult = "admitted" | "queued" | "rejected";

export class RequestQueue {
  private waiting: QueuedRequest[] = [];
  private activeCount = 0;

  constructor(
    private readonly maxConcurrent: number,
    private readonly maxQueueDepth: number,
  ) {}

  get queueDepth(): number {
    return this.waiting.length;
  }

  get active(): number {
    return this.activeCount;
  }

  // Decides what happens to a new request right now. Doesn't start
  // generation itself -- the caller does that when this returns
  // "admitted", and again later (via release()) once a queued request
  // reaches the front.
  submit(req: QueuedRequest): AdmissionResult {
    if (this.activeCount < this.maxConcurrent) {
      this.activeCount++;
      return "admitted";
    }
    if (this.waiting.length >= this.maxQueueDepth) {
      return "rejected";
    }
    this.waiting.push(req);
    return "queued";
  }

  // Call once a generation finishes, to free its slot. If anything was
  // waiting, it's admitted immediately (FIFO) and returned so the caller
  // can start generating it right away -- the same "release, then admit
  // the next thing immediately" mechanism as Fig. 2 in the blueprint,
  // just at the request level instead of the KV-cache-block level.
  release(): QueuedRequest | null {
    this.activeCount--;
    const next = this.waiting.shift();
    if (next) {
      this.activeCount++;
      return next;
    }
    return null;
  }
}
