from swarmbench import cli
from swarmbench.cli import main


def test_arena_cli(capsys) -> None:
    assert main(["arena", "--seed", "42"]) == 0
    assert "seed=42" in capsys.readouterr().out


def test_benchmark_cli(capsys) -> None:
    assert main(["benchmark", "--duration", "0.1", "--seed", "42"]) == 0
    assert "real-time" in capsys.readouterr().out


def test_render_cli_announces_replay_loading(monkeypatch, capsys, tmp_path) -> None:
    replay = object()
    output = tmp_path / "match.png"
    options = {}
    monkeypatch.setattr(cli, "load_replay", lambda _path: replay)

    def render(value, destination, **kwargs):
        options.update(kwargs)
        return destination

    monkeypatch.setattr(cli, "render_replay", render)

    assert main(["render", "match.json", "--output", str(output)]) == 0
    assert options == {"fps": 10, "quality": "low"}
    messages = capsys.readouterr().out
    assert "Loading replay from match.json..." in messages
    assert f"rendered {output}" in messages


def test_render_cli_accepts_quality_overrides(monkeypatch, tmp_path) -> None:
    options = {}
    monkeypatch.setattr(cli, "load_replay", lambda _path: object())
    monkeypatch.setattr(cli, "render_replay", lambda _replay, destination, **kwargs: options.update(kwargs) or destination)

    assert (
        main(
            [
                "render",
                "match.json",
                "--output",
                str(tmp_path / "match.mp4"),
                "--render-fps",
                "20",
                "--render-quality",
                "high",
            ]
        )
        == 0
    )
    assert options == {"fps": 20, "quality": "high"}
