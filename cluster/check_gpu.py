"""Run first, inside an interactive GPU shell, to confirm the environment
is wired up correctly before trusting any kernel benchmark numbers."""

import torch

print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"total memory: {props.total_memory / 1e9:.1f} GB")
    print(f"compute capability: {props.major}.{props.minor}")

import triton

print(f"triton: {triton.__version__}")
