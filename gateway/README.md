# gateway

Node.js + TypeScript service in front of the inference engine. Runs
against `StubBackend` (fakes token generation, no model or GPU involved)
until the real Python engine is reachable over the network — swapping it
in means writing one class that implements `Backend` and pointing
`server.ts` at it; nothing else here should need to change.

## Run it

```
npm install
npm run dev        # http://localhost:8787, auto-restarts on file changes
npm test           # unit tests for the queue, metrics, and stub backend
npm run build       # type-check + compile to dist/
```

Concurrency and queue limits are read from environment variables:
`PORT` (default 8787), `MAX_CONCURRENT` (default 4), `MAX_QUEUE_DEPTH`
(default 50).

## Try it manually

```
curl -X POST http://localhost:8787/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello", "maxTokens": 8}'
# -> {"requestId": "...", "status": "admitted", "queuePosition": 0}

curl http://localhost:8787/metrics
```

To see tokens stream, connect a WebSocket client to
`ws://localhost:8787/stream` and send
`{"type": "subscribe", "requestId": "<id from above>"}`. You'll get
`token` events, a final `done`, and a `metrics` snapshot once a second
for as long as the connection stays open.

## Design notes worth remembering for later

- **Two layers of backpressure, on purpose.** `RequestQueue` here admits
  or queues whole *requests* based on a concurrency limit. The Python
  scheduler in `engine/scheduler/` does something finer-grained: it
  admits individual *sequences* into a token-level batch as KV-cache
  blocks free up. They're solving the same kind of problem — don't let in
  more work than what's underneath can handle — at two different levels
  of the stack.
- **Late subscribers don't miss tokens.** Every request's events are
  buffered in `RequestRecord.buffer`. A WebSocket client that subscribes
  after generation has already started gets the buffer replayed first,
  then live events — so there's no race between "got a requestId back"
  and "opened the socket."
- **Finished requests are kept around for 30s** (`COMPLETED_RETENTION_MS`
  in `server.ts`) before their buffer is dropped, specifically so a
  slightly-late subscriber can still replay a request that already
  finished. This is a deliberately simple bound, not a real cache
  eviction policy — fine for a single-process gateway, worth revisiting
  if this ever needs to run as more than one instance.
