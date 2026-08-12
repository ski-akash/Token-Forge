"""A single in-flight generation request, as the scheduler sees it."""

from dataclasses import dataclass, field
from enum import Enum, auto


class SeqStatus(Enum):
    WAITING = auto()  # queued, hasn't been admitted into the batch yet
    RUNNING = auto()  # actively generating, holds at least one block
    DONE = auto()      # hit max_tokens, its blocks have been released


@dataclass
class Sequence:
    seq_id: int
    max_tokens: int
    tokens_generated: int = 0
    status: SeqStatus = SeqStatus.WAITING
    block_ids: list[int] = field(default_factory=list)

    @property
    def is_finished(self) -> bool:
        return self.tokens_generated >= self.max_tokens
