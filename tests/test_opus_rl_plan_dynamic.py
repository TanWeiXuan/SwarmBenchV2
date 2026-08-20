from __future__ import annotations

from types import SimpleNamespace

from swarmbench import DroneType

from experiments.opus_rl_plan_dynamic import _target_signature, run_instrumented_match


def test_target_signature_uses_type_and_id_without_semantic_id_features() -> None:
    target = SimpleNamespace(id=17, drone_type=DroneType.TRANSPORT)
    assert _target_signature(target) == ("TRANSPORT", 17)
    ward = SimpleNamespace(id=3, drone_type=DroneType.TRANSPORT)
    gun = SimpleNamespace(id=9, drone_type=DroneType.TANK)
    assert _target_signature((ward, gun)) == ("BLOCK_LINE", 3, 9)


def test_fixed_mode_4_effective_instrumentation_is_self_consistent() -> None:
    result = run_instrumented_match(
        opponent="potential",
        seed=3_100_003,
        side="A",
        subject="fixed_mode_4",
        duration=2.0,
    )
    assert result["effective_decisions"] == 2
    assert result["scout_command_differences_from_mode_4"] == 0
    assert result["role_timeline"][0]["duties"]
