"""Week 1, sanity check: a trivial Triton kernel to prove the pipeline works
(Slurm allocation -> CUDA -> torch -> Triton JIT compilation) before writing
kernels that actually matter (quantized matmul, fused softmax).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape and x.is_cuda and y.is_cuda
    out = torch.empty_like(x)
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _vector_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


def _check_correctness():
    torch.manual_seed(0)
    x = torch.rand(2**20, device="cuda")
    y = torch.rand(2**20, device="cuda")
    triton_out = vector_add(x, y)
    torch_out = x + y
    max_diff = (triton_out - torch_out).abs().max().item()
    print(f"max diff vs torch: {max_diff:.2e}")
    assert torch.allclose(triton_out, torch_out)
    print("correctness: OK")


def _benchmark(n_elements: int = 2**26, iters: int = 50):
    x = torch.rand(n_elements, device="cuda")
    y = torch.rand(n_elements, device="cuda")

    def run(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters

    triton_ms = run(lambda: vector_add(x, y))
    torch_ms = run(lambda: x + y)

    def gbps(ms):
        # read x, read y, write out = 3 tensors moved
        return 3 * n_elements * x.element_size() / ms * 1e-6

    print(f"triton: {triton_ms:.4f} ms  ({gbps(triton_ms):.1f} GB/s)")
    print(f"torch:  {torch_ms:.4f} ms  ({gbps(torch_ms):.1f} GB/s)")


if __name__ == "__main__":
    _check_correctness()
    _benchmark()
