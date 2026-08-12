"""Unit tests for the block pool itself, independent of the scheduler."""

import pytest

from engine.scheduler.block_allocator import BlockAllocator, OutOfBlocksError


def test_allocate_reduces_free_count():
    pool = BlockAllocator(num_blocks=4)
    blocks = pool.allocate(2)
    assert len(blocks) == 2
    assert pool.num_free == 2


def test_allocate_more_than_available_raises():
    pool = BlockAllocator(num_blocks=2)
    with pytest.raises(OutOfBlocksError):
        pool.allocate(3)


def test_free_returns_blocks_to_the_pool():
    pool = BlockAllocator(num_blocks=4)
    blocks = pool.allocate(3)
    pool.free(blocks)
    assert pool.num_free == 4


def test_blocks_are_never_double_allocated():
    pool = BlockAllocator(num_blocks=4)
    first = pool.allocate(2)
    second = pool.allocate(2)
    assert set(first).isdisjoint(second)
