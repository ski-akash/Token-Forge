// The gateway doesn't know or care how tokens actually get produced -- it
// only depends on this interface. Today only StubBackend implements it,
// faking generation so the whole gateway (queue, WebSocket streaming,
// metrics, backpressure) can be built and tested without the Python
// inference engine or a GPU. Once that engine exists and exposes an HTTP
// or gRPC endpoint, a class that calls it implements this same interface
// and drops in without touching server.ts or the queue at all.

export interface Backend {
  // Yields one token at a time and stops after `maxTokens` tokens.
  generate(prompt: string, maxTokens: number): AsyncGenerator<string>;
}
