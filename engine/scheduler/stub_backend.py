"""
Fake token-generation backend, used only so the scheduler can be built
and tested without a GPU. It doesn't run a model at all -- it just
pretends a forward pass happened, which is enough for the scheduler,
since scheduling logic (who runs, who waits, who gets which blocks)
doesn't actually depend on what the model computes.

Once the real Triton-kernel-backed engine exists, it should implement
this same generate_token signature and drop in as a straight replacement
-- scheduler.py should not need to change at all when that happens.
"""


class StubBackend:
    def generate_token(self, seq_id: int, tokens_so_far: int) -> None:
        # Nothing to compute -- the scheduler only needs to know this
        # call happened so it can advance the sequence's token count.
        return None
