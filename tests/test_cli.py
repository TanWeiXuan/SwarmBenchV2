from types import SimpleNamespace

from swarmbench import Team
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
    assert options == {
        "fps": 10,
        "quality": "low",
        "killfeed": False,
        "killfeed_duration": 5.0,
        "killfeed_lines": 5,
    }
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
                "--killfeed",
                "--killfeed-duration",
                "7.5",
                "--killfeed-lines",
                "3",
            ]
        )
        == 0
    )
    assert options == {
        "fps": 20,
        "quality": "high",
        "killfeed": True,
        "killfeed_duration": 7.5,
        "killfeed_lines": 3,
    }


def test_match_cli_passes_killfeed_options_to_renderer(monkeypatch, tmp_path) -> None:
    replay = object()
    output = tmp_path / "match.mp4"
    options = {}
    result = SimpleNamespace(score_a=1, score_b=0, winner=Team.A, replay=replay)
    monkeypatch.setattr(cli, "run_match", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(
        cli,
        "render_replay",
        lambda value, destination, **kwargs: options.update(kwargs) or destination,
    )

    assert (
        main(
            [
                "match",
                "--controller-a",
                "rush",
                "--controller-b",
                "defend",
                "--seed",
                "42",
                "--render",
                str(output),
                "--killfeed",
            ]
        )
        == 0
    )
    assert options == {
        "fps": 10,
        "quality": "low",
        "killfeed": True,
        "killfeed_duration": 5.0,
        "killfeed_lines": 5,
    }
