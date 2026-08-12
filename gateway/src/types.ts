// Shared request/response shapes for the gateway. Kept in one file so the
// REST handlers, the WebSocket layer, and the queue all agree on the same
// wire format instead of each defining their own slightly-different copy.

export interface GenerateRequestBody {
  prompt: string;
  maxTokens: number;
}

export interface QueuedRequest {
  id: string;
  prompt: string;
  maxTokens: number;
  enqueuedAt: number;
}

// Messages pushed to WebSocket clients subscribed to a given request.
export type StreamEvent =
  | { type: "token"; requestId: string; token: string }
  | { type: "done"; requestId: string; tokensGenerated: number }
  | { type: "error"; requestId: string; message: string };

export interface MetricsSnapshot {
  activeGenerations: number;
  queueDepth: number;
  totalRequestsCompleted: number;
  totalTokensGenerated: number;
  tokensPerSecond: number;
}
