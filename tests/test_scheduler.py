"""
test_admits_waiting_sequence_as_soon_as_blocks_free_up is the executable
version of Fig. 2 in the project blueprint: a finishing sequence's block
gets reused by a newly admitted sequence on the very next step, instead
of that sequence sitting in the queue until every other running sequence
also finishes.
"""

import pytest

from engine.scheduler.block_allocator import OutOfBlocksError
from engine.scheduler.scheduler import Scheduler
from engine.scheduler.sequence import Sequence, SeqStatus
from engine.scheduler.stub_backend import StubBackend
from engine.scheduler.block_allocator import BlockAllocator


def make_scheduler(num_blocks: int, block_size: int) -> Scheduler:
    return Scheduler(
        allocator=BlockAllocator(num_blocks=num_blocks),
        backend=StubBackend(),
        block_size=block_size,
    )


def test_admits_waiting_sequence_as_soon_as_blocks_free_up():
    # Pool only has 2 blocks (4 tokens each), so only two sequences can be
    # admitted up front. A is short and finishes fast; B is long-running.
    # C should have to wait for A, but NOT for B.
    sched = make_scheduler(num_blocks=2, block_size=4)
    seq_a = Sequence(seq_id=1, max_tokens=2)
    seq_b = Sequence(seq_id=2, max_tokens=4)
    seq_c = Sequence(seq_id=3, max_tokens=2)
    sched.submit(seq_a)
    sched.submit(seq_b)
    sched.submit(seq_c)

    # Step 1: A and B take the only 2 blocks. C has nowhere to go yet.
    result = sched.step()
    assert set(result.admitted) == {1, 2}
    assert seq_c.status == SeqStatus.WAITING

    # Step 2: A reaches max_tokens and finishes, releasing its block.
    # B is still only halfway done.
    result = sched.step()
    assert result.finished == [1]
    assert seq_b.status == SeqStatus.RUNNING

    # Step 3: C gets admitted immediately using A's freed block -- it does
    # not sit in the queue until B (still 2 tokens away) finishes too.
    # This is the actual behavior Fig. 2 in the blueprint claims.
    result = sched.step()
    assert 3 in result.admitted
    assert seq_c.status == SeqStatus.RUNNING
    assert 2 not in result.finished  # B is still running


def test_sequence_grows_into_a_second_block_then_releases_both():
    sched = make_scheduler(num_blocks=3, block_size=2)
    seq = Sequence(seq_id=1, max_tokens=4)  # needs 2 blocks by the end
    sched.submit(seq)

    sched.step()  # admitted + 1st token -- 1 block, half used
    sched.step()  # 2nd token -- still fits in that 1 block
    assert len(seq.block_ids) == 1

    sched.step()  # 3rd token -- doesn't fit anymore, grabs a 2nd block
    assert len(seq.block_ids) == 2

    result = sched.step()  # 4th token -> hits max_tokens, finishes
    assert result.finished == [1]
    assert sched.allocator.num_free == 3  # both blocks back in the pool


def test_growth_failure_is_not_silently_swallowed():
    # If a running sequence needs a new block and the pool is empty,
    # that's a real problem a production scheduler would solve with
    # preemption. Not implemented yet -- so today this should fail loudly
    # rather than corrupt state or silently stall the sequence.
    sched = make_scheduler(num_blocks=1, block_size=2)
    seq = Sequence(seq_id=1, max_tokens=4)  # will eventually need 2 blocks
    sched.submit(seq)

    sched.step()  # takes the only block
    sched.step()  # still fits in it
    with pytest.raises(OutOfBlocksError):
        sched.step()  # needs a 2nd block, pool is empty
