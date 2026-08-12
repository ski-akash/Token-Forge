import { describe, expect, it } from "vitest";
import { StubBackend } from "../src/stubBackend";

describe("StubBackend", () => {
  it("yields exactly maxTokens tokens then stops", async () => {
    const backend = new StubBackend(1); // 1ms/token so the test stays fast
    const tokens: string[] = [];
    for await (const token of backend.generate("hello", 5)) {
      tokens.push(token);
    }
    expect(tokens).toHaveLength(5);
  });
});
