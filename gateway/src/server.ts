import { randomUUID } from "crypto";
import { createServer } from "http";
import { EventEmitter } from "events";
import express from "express";
import { WebSocketServer, WebSocket } from "ws";

import { StubBackend } from "./stubBackend";
import { RequestQueue } from "./requestQueue";
import { Metrics } from "./metrics";
import { GenerateRequestBody, QueuedRequest, StreamEvent } from "./types";

const PORT = Number(process.env.PORT ?? 8787);
const MAX_CONCURRENT = Number(process.env.MAX_CONCURRENT ?? 4);
const MAX_QUEUE_DEPTH = Number(process.env.MAX_QUEUE_DEPTH ?? 50);
// How long a finished request's token buffer stays around, so a client
// that connects to the WebSocket a little late (there's always some gap
// between a client getting a requestId back and it opening the socket)
// can still replay everything it missed instead of getting nothing.
const COMPLETED_RETENTION_MS = 30_000;

const backend = new StubBackend();
const queue = new RequestQueue(MAX_CONCURRENT, MAX_QUEUE_DEPTH);
const metrics = new Metrics();

interface RequestRecord {
  emitter: EventEmitter;
  // Every event this request has ever emitted, in order. A client that
  // subscribes after generation has already started replays this buffer
  // first, then gets live events -- so "subscribed late" never means
  // "missed tokens".
  buffer: StreamEvent[];
  done: boolean;
}

const requests = new Map<string, RequestRecord>();

function emitEvent(record: RequestRecord, event: StreamEvent): void {
  record.buffer.push(event);
  record.emitter.emit("event", event);
}

function startGeneration(req: QueuedRequest): void {
  const record: RequestRecord = { emitter: new EventEmitter(), buffer: [], done: false };
  requests.set(req.id, record);

  (async () => {
    let tokensGenerated = 0;
    try {
      for await (const token of backend.generate(req.prompt, req.maxTokens)) {
        tokensGenerated++;
        metrics.recordToken();
        emitEvent(record, { type: "token", requestId: req.id, token });
      }
      emitEvent(record, { type: "done", requestId: req.id, tokensGenerated });
    } catch (err) {
      emitEvent(record, {
        type: "error",
        requestId: req.id,
        message: err instanceof Error ? err.message : "generation failed",
      });
    } finally {
      metrics.recordRequestCompleted();
      record.done = true;
      // Keep the record around briefly for late subscribers, then let it
      // get garbage collected -- otherwise every request served would
      // leak memory forever.
      setTimeout(() => requests.delete(req.id), COMPLETED_RETENTION_MS);

      // A slot just freed up. If anything was waiting, start it
      // immediately -- this is the actual backpressure/admission
      // behavior under test in requestQueue.test.ts.
      const next = queue.release();
      if (next) startGeneration(next);
    }
  })();
}

const app = express();
app.use(express.json());

app.post("/generate", (req, res) => {
  const body = req.body as Partial<GenerateRequestBody>;
  if (typeof body.prompt !== "string" || body.prompt.trim().length === 0) {
    res.status(400).json({ error: "prompt is required" });
    return;
  }
  const maxTokens = typeof body.maxTokens === "number" && body.maxTokens > 0 ? body.maxTokens : 32;

  const id = randomUUID();
  const queuedRequest: QueuedRequest = { id, prompt: body.prompt, maxTokens, enqueuedAt: Date.now() };
  const result = queue.submit(queuedRequest);

  if (result === "rejected") {
    res.status(503).json({ error: "queue is full, try again shortly" });
    return;
  }
  if (result === "admitted") {
    startGeneration(queuedRequest);
  }
  // "queued" requests are picked up later, inside startGeneration's
  // finally block above, once an earlier request finishes and releases
  // its slot.

  res.status(202).json({ requestId: id, status: result, queuePosition: queue.queueDepth });
});

app.get("/metrics", (_req, res) => {
  res.json(metrics.snapshot(queue.active, queue.queueDepth));
});

const httpServer = createServer(app);
const wss = new WebSocketServer({ server: httpServer, path: "/stream" });

wss.on("connection", (ws: WebSocket) => {
  // Track this socket's own listeners so they can be detached on close
  // instead of leaking a reference on every finished request's emitter.
  const subscriptions = new Map<string, (event: StreamEvent) => void>();

  ws.on("message", (raw) => {
    let msg: { type?: string; requestId?: string };
    try {
      msg = JSON.parse(raw.toString());
    } catch {
      return;
    }
    if (msg.type !== "subscribe" || typeof msg.requestId !== "string") {
      return;
    }

    const record = requests.get(msg.requestId);
    if (!record) {
      ws.send(JSON.stringify({ type: "error", requestId: msg.requestId, message: "unknown or expired request" }));
      return;
    }

    // Replay everything that already happened, then attach a live
    // listener for anything still to come. Both happen synchronously
    // here (no `await` in between), so nothing emitted after the buffer
    // read can slip through before the listener is attached.
    for (const event of record.buffer) {
      ws.send(JSON.stringify(event));
    }
    if (!record.done) {
      const listener = (event: StreamEvent) => ws.send(JSON.stringify(event));
      record.emitter.on("event", listener);
      subscriptions.set(msg.requestId, listener);
    }
  });

  // Push a metrics snapshot every second so a connected dashboard sees
  // live numbers without having to poll GET /metrics itself.
  const metricsInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "metrics", ...metrics.snapshot(queue.active, queue.queueDepth) }));
    }
  }, 1000);

  ws.on("close", () => {
    clearInterval(metricsInterval);
    for (const [requestId, listener] of subscriptions) {
      requests.get(requestId)?.emitter.off("event", listener);
    }
  });
});

httpServer.listen(PORT, () => {
  console.log(`gateway listening on http://localhost:${PORT}`);
});
