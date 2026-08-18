"""Controller subprocess entry point. Protocol output is isolated from prints."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any

from swarmbench.api import BaseSwarmController

from .protocol import PROTOCOL_VERSION, game_info_from_dict, game_state_from_dict, validate_request


def _seed_libraries(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass


def _seed_torch_if_loaded(seed: int) -> None:
    """Seed PyTorch without imposing its import cost on controllers that do not use it."""
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(int(os.environ.get("SWARMBENCH_TORCH_THREADS", "1")))
        torch.use_deterministic_algorithms(True, warn_only=True)


def _load_controller(path: Path, seed: int) -> BaseSwarmController:
    spec = importlib.util.spec_from_file_location("swarmbench_submission", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load controller: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    controller_type = getattr(module, "SwarmController", None)
    if not isinstance(controller_type, type) or not issubclass(controller_type, BaseSwarmController):
        raise TypeError("controller file must define SwarmController(BaseSwarmController)")
    _seed_torch_if_loaded(seed)
    return controller_type()


def _response(sequence: int, status: str, **payload: Any) -> dict[str, Any]:
    return {"protocol_version": PROTOCOL_VERSION, "sequence": sequence, "status": status, **payload}


def main() -> int:
    protocol_out = sys.stdout
    seed = int(os.environ["SWARMBENCH_CONTROLLER_SEED"])
    try:
        with contextlib.redirect_stdout(sys.stderr):
            _seed_libraries(seed)
            controller = _load_controller(Path(sys.argv[1]).resolve(), seed)
    except BaseException:
        print(json.dumps(_response(0, "error", error=traceback.format_exc(limit=20))), file=protocol_out, flush=True)
        return 1

    for line in sys.stdin:
        sequence = -1
        try:
            message = validate_request(json.loads(line))
            sequence = message["sequence"]
            command = message["command"]
            with contextlib.redirect_stdout(sys.stderr):
                if command == "initialize":
                    controller.initialize(game_info_from_dict(message["payload"]))
                    result: Any = None
                elif command == "step":
                    result = controller.step(game_state_from_dict(message["payload"]))
                else:
                    print(json.dumps(_response(sequence, "ok", result=None)), file=protocol_out, flush=True)
                    return 0
            print(json.dumps(_response(sequence, "ok", result=result), allow_nan=True), file=protocol_out, flush=True)
        except BaseException:
            print(json.dumps(_response(sequence, "error", error=traceback.format_exc(limit=20))), file=protocol_out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
