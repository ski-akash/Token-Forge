# Inference Stack

Custom LLM fine-tuning and serving platform: continuous-batching inference
engine with hand-written Triton kernels and paged KV-cache, fronted by a
TypeScript control plane. Build plan and architecture diagrams:
see the published blueprint (linked in project notes).

## Layout

- `engine/kernels/` — Triton kernels, one file per kernel, each with a
  correctness check and a benchmark in `if __name__ == "__main__"`.
- `engine/scheduler/` — continuous-batching scheduler and the paged
  KV-cache block allocator. Pure Python, no GPU required to build or
  test: the scheduler talks to a `Backend` interface, and `StubBackend`
  fakes token generation so this can be developed and tested without
  cluster access. The real Triton-kernel-backed engine implements the
  same interface later and should drop in without changing scheduler.py.
- `cluster/` — everything needed to reproduce the environment and get an
  interactive GPU shell on the Slurm cluster.
- `tests/` — run with `python -m pytest` from the repo root.

## Step 1: verify the environment on the cluster

1. Copy this repo to the cluster (git clone, or `scp -r`).
2. On the login node, run `bash cluster/env_setup.sh` once. It creates a
   venv at `.venv` and installs torch + triton. **Edit the CUDA module name
   and the torch index URL first** — run `module avail cuda` and
   `nvidia-smi` to find the right versions for this cluster.
3. Edit `cluster/interactive_gpu.sh` — fill in `--partition` (and
   `--account` if required) with values from `sinfo` and
   `sacctmgr show associations -p`. Ask the HPC helpdesk if it's unclear
   which partition has the A100s.
4. Get an interactive GPU shell: `bash cluster/interactive_gpu.sh`.
5. Inside that shell: `source .venv/bin/activate && python cluster/check_gpu.py`.
   This should print the GPU name, its memory, and the installed
   torch/triton versions.
6. Still inside that shell: `python engine/kernels/vector_add.py`. This
   compiles and runs a real (if trivial) Triton kernel, checks it against
   `torch`, and prints achieved memory bandwidth. If this passes, the
   pipeline — Slurm allocation, CUDA, torch, Triton compilation — is
   confirmed working, and week 1's real kernels (quantized matmul, fused
   softmax) can build on top of it.

## Status

- [ ] Environment + kernel-pipeline sanity check (pending cluster access)
- [ ] Quantized matmul kernel
- [ ] Fused softmax kernel
- [ ] LoRA fine-tuning pipeline
- [ ] Naive serving baseline
- [x] Continuous batching scheduler (built + unit-tested off-cluster, no GPU needed)
- [~] Paged KV-cache (block allocator done off-cluster; kernel integration still pending cluster access)
- [ ] Benchmark harness vs vLLM
- [ ] Node.js/TypeScript gateway
- [ ] React/TypeScript dashboard
