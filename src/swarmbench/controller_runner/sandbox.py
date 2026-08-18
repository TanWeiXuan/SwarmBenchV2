"""Docker command construction for authoritative untrusted execution."""

from __future__ import annotations

from pathlib import Path


def docker_run_command(image: str, controller: Path, seed: int) -> list[str]:
    """Build a networkless, read-only, resource-limited controller command."""
    return [
        "docker",
        "run",
        "--rm",
        "-i",
        "--network",
        "none",
        "--read-only",
        "--cpus",
        "1",
        "--memory",
        "2g",
        "--pids-limit",
        "128",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=128m",
        "--env",
        f"SWARMBENCH_CONTROLLER_SEED={seed}",
        "--env",
        f"PYTHONHASHSEED={seed % (2**32)}",
        "--env",
        "OMP_NUM_THREADS=1",
        "--env",
        "MKL_NUM_THREADS=1",
        "--env",
        "OPENBLAS_NUM_THREADS=1",
        "--env",
        "SWARMBENCH_TORCH_THREADS=1",
        "--env",
        "CUDA_VISIBLE_DEVICES=",
        "--mount",
        f"type=bind,src={controller.resolve()},dst=/submission/controller.py,readonly",
        image,
        "python",
        "-u",
        "-m",
        "swarmbench.controller_runner.worker",
        "/submission/controller.py",
    ]
