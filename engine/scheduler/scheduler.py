"""
Continuous-batching scheduler.

The core idea (Fig. 2 in the project blueprint): don't run a fixed batch
until every sequence in it finishes. Instead, admit a new waiting request
into the batch on the very next step, as soon as a finished sequence's
blocks free up. That's what keeps the GPU busy instead of idling while
one slow sequence in the batch drags behind its already-finished
siblings.

Block accounting is on-demand, not reserved up front: a sequence starts
with one block and only grabs another once it actually runs out of room
in the ones it already has. This mirrors how paged attention manages KV
cache memory for real.

Known simplification: if a running sequence needs a new block and the
pool is empty, this currently raises rather than handling it gracefully.
A production scheduler would preempt a lower-priority sequence (evict its
blocks back to the pool, re-admit it later) instead of failing. That's a
natural next step, not implemented yet -- see test_growth_failure_is_not_silently_swallowed
in tests/test_scheduler.py, which pins down today's behavior on purpose.
"""

from dataclasses import dataclass

from engine.scheduler.block_allocator import BlockAllocator
from engine.scheduler.sequence import Sequence, SeqStatus


@dataclass
class StepResult:
    admitted: list[int]
    generated: list[int]
    finished: list[int]


class Scheduler:
    def __init__(self, allocator: BlockAllocator, backend, block_size: int):
        self.allocator = allocator
        self.backend = backend
        self.block_size = block_size
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []

    def submit(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def _blocks_needed(self, tokens_generated: int) -> int:
        # A sequence with 5 tokens and a block size of 2 needs 3 blocks:
        # 2 full ones plus 1 holding the leftover token. Round up.
        if tokens_generated == 0:
            return 1
        return -(-tokens_generated // self.block_size)

    def _try_admit(self) -> list[int]:
        """Pull waiting sequences into the running batch while blocks last.

        Admission is FIFO and only grants the one block a sequence needs
        to start -- it does not reserve blocks for a sequence's eventual
        max_tokens. That's what lets a short sequence and a long one share
        the pool without the long one blocking everyone else up front.
        """
        admitted = []
        while self.waiting and self.allocator.num_free >= 1:
            seq = self.waiting[0]
            seq.block_ids = self.allocator.allocate(1)
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            self.waiting.pop(0)
            admitted.append(seq.seq_id)
        return admitted

    def step(self) -> StepResult:
        """Run one decode iteration: admit, generate one token per running
        sequence, grow blocks as needed, release blocks for anything that
        just finished."""
        admitted = self._try_admit()

        generated: list[int] = []
        finished: list[int] = []
        still_running: list[Sequence] = []

        for seq in self.running:
            self.backend.generate_token(seq.seq_id, seq.tokens_generated)
            seq.tokens_generated += 1
            generated.append(seq.seq_id)

            needed = self._blocks_needed(seq.tokens_generated)
            if needed > len(seq.block_ids):
                # Just crossed a block boundary -- grab one more. Raises
                # if the pool is empty; see the module docstring.
                seq.block_ids += self.allocator.allocate(needed - len(seq.block_ids))

            if seq.is_finished:
                seq.status = SeqStatus.DONE
                self.allocator.free(seq.block_ids)
                finished.append(seq.seq_id)
            else:
                still_running.append(seq)

        self.running = still_running
        return StepResult(admitted=admitted, generated=generated, finished=finished)
