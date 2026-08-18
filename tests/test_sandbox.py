from pathlib import Path

from swarmbench.controller_runner.sandbox import docker_run_command


def test_docker_command_has_required_isolation(tmp_path: Path) -> None:
    controller = tmp_path / "controller.py"
    controller.write_text("# controller", encoding="utf-8")
    command = docker_run_command("swarmbench-controller", controller, 123)
    assert command[:3] == ["docker", "run", "--rm"]
    assert ["--network", "none"] == command[command.index("--network") : command.index("--network") + 2]
    assert "--read-only" in command
    assert "--pids-limit" in command
    assert any("readonly" in argument and "controller.py" in argument for argument in command)
    assert not any("GITHUB" in argument for argument in command)
