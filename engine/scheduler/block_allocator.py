"""
Fixed-size block pool for the KV cache.

Real inference engines don't reserve a full max_length worth of KV cache
per sequence up front -- that wastes GPU memory, since most sequences
never actually hit max_length. Instead the cache is split into fixed-size
blocks, and a sequence only grabs a new one when it actually needs it.
This file is the pure bookkeeping side of that idea: no tensors, no CUDA,
just tracking which block ids are currently free. It's meant to be usable
and testable on a laptop; a real GPU-backed KV-cache tensor gets indexed
by the same block ids later, once that part exists.
"""

from dataclasses import dataclass, field


class OutOfBlocksError(RuntimeError):
    """Raised when the pool has no free blocks left to hand out."""


@dataclass
class BlockAllocator:
    num_blocks: int
    _free: list[int] = field(init=False)

    def __post_init__(self) -> None:
        # Block ids are just indices into the (eventually real) KV-cache
        # tensor. Order doesn't matter for correctness, so a plain list
        # used as a stack is enough -- no need for a fancier free-list
        # structure at this scale.
        self._free = list(range(self.num_blocks))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, count: int = 1) -> list[int]:
        """Grab `count` blocks, or raise if the pool doesn't have that many.

        Every caller here allocates for a single sequence at once, so an
        all-or-nothing allocation is fine -- there's no case yet where a
        partial grant would be useful.
        """
        if count > self.num_free:
            raise OutOfBlocksError(
                f"requested {count} blocks, only {self.num_free} free"
            )
        blocks, self._free = self._free[:count], self._free[count:]
        return blocks

    def free(self, block_ids: list[int]) -> None:
        """Return blocks to the pool, e.g. once a sequence finishes."""
        self._free.extend(block_ids)
