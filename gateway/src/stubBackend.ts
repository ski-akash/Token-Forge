import { Backend } from "./backend";

// A small fixed vocabulary so streamed output looks like words instead of
// "token_1 token_2 ...". The content is meaningless -- this backend exists
// to exercise streaming and timing behavior, not to generate real text.
const FAKE_VOCAB = [
  "the", "model", "is", "generating", "a", "response", "one", "token",
  "at", "a", "time", "while", "the", "scheduler", "keeps", "the", "batch",
  "full", "and", "the", "gateway", "streams", "results", "back",
];

export class StubBackend implements Backend {
  constructor(private readonly msPerToken = 60) {}

  async *generate(prompt: string, maxTokens: number): AsyncGenerator<string> {
    for (let i = 0; i < maxTokens; i++) {
      // A real backend's delay here is a GPU forward pass; this just
      // sleeps, so nothing downstream (the queue, the WS layer, the
      // metrics) can tell the difference between "waiting on a model"
      // and "waiting on a timer".
      await new Promise((resolve) => setTimeout(resolve, this.msPerToken));
      yield FAKE_VOCAB[(prompt.length + i) % FAKE_VOCAB.length];
    }
  }
}
