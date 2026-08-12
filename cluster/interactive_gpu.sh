#!/usr/bin/env bash
# Request an interactive shell with one GPU, for iterative kernel dev
# (as opposed to sbatch, which is for unattended long-running jobs like
# training, added in a later week).
#
# EDIT BEFORE RUNNING:
#   --partition : run `sinfo` and pick the partition that lists A100s.
#   --account   : run `sacctmgr show associations -p` if this cluster
#                 requires an account/allocation to be specified.
srun --partition=<gpu-partition> \
     --gres=gpu:1 \
     --cpus-per-task=8 \
     --mem=32G \
     --time=04:00:00 \
     --pty bash
