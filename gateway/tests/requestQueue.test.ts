import { describe, expect, it } from "vitest";
import { RequestQueue } from "../src/requestQueue";
import { QueuedRequest } from "../src/types";

function makeRequest(id: string): QueuedRequest {
  return { id, prompt: "hi", maxTokens: 8, enqueuedAt: Date.now() };
}

describe("RequestQueue", () => {
  it("admits requests immediately while under the concurrency limit", () => {
    const q = new RequestQueue(2, 10);
    expect(q.submit(makeRequest("a"))).toBe("admitted");
    expect(q.submit(makeRequest("b"))).toBe("admitted");
    expect(q.active).toBe(2);
  });

  it("queues requests once the concurrency limit is hit", () => {
    const q = new RequestQueue(1, 10);
    expect(q.submit(makeRequest("a"))).toBe("admitted");
    expect(q.submit(makeRequest("b"))).toBe("queued");
    expect(q.queueDepth).toBe(1);
  });

  it("rejects new requests once the queue itself is full", () => {
    const q = new RequestQueue(1, 1);
    q.submit(makeRequest("a")); // admitted, takes the only slot
    q.submit(makeRequest("b")); // queued, takes the only queue spot
    expect(q.submit(makeRequest("c"))).toBe("rejected");
  });

  it("admits the next waiting request as soon as a slot frees, FIFO", () => {
    // This is the request-level version of the Fig. 2 mechanism: "a"
    // finishing should immediately free its slot for "b", without "b"
    // having to wait for anything else.
    const q = new RequestQueue(1, 10);
    q.submit(makeRequest("a"));
    q.submit(makeRequest("b"));
    q.submit(makeRequest("c"));

    const next = q.release(); // "a" finishes
    expect(next?.id).toBe("b");
    expect(q.active).toBe(1);
    expect(q.queueDepth).toBe(1); // "c" is still waiting
  });

  it("returns null from release when nothing is waiting", () => {
    const q = new RequestQueue(2, 10);
    q.submit(makeRequest("a"));
    expect(q.release()).toBeNull();
    expect(q.active).toBe(0);
  });
});
