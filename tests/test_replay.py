import io
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from swarmbench.api import DroneStatus, DroneType, Team
from swarmbench.controllers.baselines import baseline_path
from swarmbench.engine import generate_scenario
from swarmbench.match import run_match
from swarmbench.replay import ReplayValidationError, load_replay, reconstruct_frames, save_replay, validate_replay, verify_reconstruction
from swarmbench.replay import renderer
from swarmbench.replay.renderer import explosion_events_at, render_arena, render_replay


@pytest.fixture(scope="module")
def short_match():
    return run_match(baseline_path("rush"), baseline_path("defend"), seed=12, duration=1.0)


def test_replay_save_load_round_trip(short_match, tmp_path: Path) -> None:
    destination = save_replay(short_match.replay, tmp_path / "match.json")
    loaded = load_replay(destination)
    assert loaded.to_dict() == short_match.replay.to_dict()
    verify_reconstruction(loaded)


def test_compressed_replay_round_trip(short_match, tmp_path: Path) -> None:
    destination = save_replay(short_match.replay, tmp_path / "match.json.gz")
    assert load_replay(destination).to_dict() == short_match.replay.to_dict()


def test_replay_validation_rejects_wrong_version(short_match) -> None:
    data = short_match.replay.to_dict()
    data["replay_version"] = 999
    with pytest.raises(ReplayValidationError):
        validate_replay(data)


def test_reconstruction_is_deterministic(short_match) -> None:
    first = list(reconstruct_frames(short_match.replay))
    second = list(reconstruct_frames(short_match.replay))
    assert first == second


def test_renderer_creates_image(short_match, tmp_path: Path, capsys) -> None:
    from PIL import Image

    output = render_replay(short_match.replay, tmp_path / "match.png")
    assert output is not None and output.stat().st_size > 1_000
    with Image.open(output) as image:
        assert image.size == (640, 384)
    messages = capsys.readouterr().out
    assert "Reconstructing replay frames..." in messages
    assert "Prepared 11 frames at 10 FPS (low quality)." in messages
    assert "Rendering final frame" in messages
    assert "Finished rendering" in messages


def test_renderer_creates_high_quality_jpeg(short_match, tmp_path: Path) -> None:
    from PIL import Image

    output = render_replay(short_match.replay, tmp_path / "match.jpg", quality="high")
    with Image.open(output) as image:
        assert image.size == (1400, 840)
        assert image.format == "JPEG"


def test_renderer_header_uses_controller_names(short_match, tmp_path: Path, monkeypatch) -> None:
    fitted_values = []
    fit_text = renderer._fit_text

    def record_fitted_value(draw, value, max_width, font):
        fitted_values.append(value)
        return fit_text(draw, value, max_width, font)

    monkeypatch.setattr(renderer, "_fit_text", record_fitted_value)
    render_replay(short_match.replay, tmp_path / "match.png")

    assert fitted_values == ["Blue A: rush", "Red B: defend"]


def test_renderer_header_counts_only_active_drones(short_match) -> None:
    frame = next(reconstruct_frames(short_match.replay))
    expected = tuple(
        sum(drone.team is Team.A and drone.drone_type is kind for drone in frame.drones)
        for kind in DroneType
    )
    assert renderer._remaining_counts(frame, Team.A) == expected
    assert renderer._remaining_counts(frame, Team.B) == expected

    removed = next(
        drone for drone in frame.drones if drone.team is Team.A and drone.drone_type is DroneType.SCOUT
    )
    changed = replace(
        frame,
        drones=tuple(
            replace(drone, status=DroneStatus.ELIMINATED) if drone.id == removed.id else drone
            for drone in frame.drones
        ),
    )
    assert renderer._remaining_counts(changed, Team.A) == (expected[0] - 1, expected[1], expected[2])
    assert renderer._count_text(expected) == f"S {expected[0]}  Tr {expected[1]}  Tk {expected[2]}"


def test_killfeed_uses_vehicle_icons_without_names_or_ids(short_match) -> None:
    replay = deepcopy(short_match.replay)
    red_tank = next(drone for drone in replay.scenario.drones if drone.team is Team.B and drone.drone_type is DroneType.TANK)
    blue_transport = next(
        drone for drone in replay.scenario.drones if drone.team is Team.A and drone.drone_type is DroneType.TRANSPORT
    )
    red_scouts = [
        drone for drone in replay.scenario.drones if drone.team is Team.B and drone.drone_type is DroneType.SCOUT
    ][:2]
    replay.events.extend(
        [
            {
                "time": 0.2,
                "type": "PROJECTILE_FIRED",
                "drone_ids": [red_tank.id],
                "position": [0.0, 0.0],
                "team": "B",
                "points": 0,
                "projectile_id": 99,
            },
            {
                "time": 0.5,
                "type": "PROJECTILE_HIT",
                "drone_ids": [blue_transport.id],
                "position": [0.0, 0.0],
                "team": "B",
                "points": 0,
                "projectile_id": 99,
            },
            {
                "time": 0.6,
                "type": "VEHICLE_COLLISION",
                "drone_ids": [red_scouts[0].id, red_scouts[1].id],
                "position": [0.0, 0.0],
                "team": None,
                "points": 0,
            },
            {
                "time": 0.7,
                "type": "GOAL",
                "drone_ids": [blue_transport.id],
                "position": [0.0, 0.0],
                "team": "A",
                "points": 5,
            },
        ]
    )

    entries = renderer._killfeed_entries(replay)

    assert entries[-3].kind == "hit"
    assert entries[-3].vehicles == ((Team.B, DroneType.TANK), (Team.A, DroneType.TRANSPORT))
    assert entries[-2].kind == "collision"
    assert entries[-2].vehicles == ((Team.B, DroneType.SCOUT), (Team.B, DroneType.SCOUT))
    assert entries[-1].kind == "score"
    assert entries[-1].vehicles == ((Team.A, DroneType.TRANSPORT),)
    assert entries[-1].points == 5


def test_killfeed_defaults_to_five_recent_seconds_and_lines(short_match) -> None:
    vehicle = (Team.A, DroneType.SCOUT)
    entries = tuple(renderer._KillfeedEntry(float(index), "score", (vehicle,), 1) for index in range(7))

    visible = renderer._visible_killfeed_entries(entries, 6.0, duration=5.0, limit=5)

    assert [entry.time for entry in visible] == [6.0, 5.0, 4.0, 3.0, 2.0]
    assert not renderer._visible_killfeed_entries(entries, 12.0, duration=5.0, limit=5)


def test_renderer_draws_icon_killfeed(short_match, tmp_path: Path) -> None:
    replay = deepcopy(short_match.replay)
    scorer = next(drone for drone in replay.scenario.drones if drone.drone_type is DroneType.TRANSPORT)
    replay.events.append(
        {
            "time": 0.9,
            "type": "GOAL",
            "drone_ids": [scorer.id],
            "position": list(scorer.position),
            "team": scorer.team.value,
            "points": 5,
        }
    )

    output = render_replay(replay, tmp_path / "killfeed.png", killfeed=True)

    assert output.stat().st_size > 1_000


def test_renderer_truncates_long_controller_names() -> None:
    from PIL import Image, ImageDraw, ImageFont

    draw = ImageDraw.Draw(Image.new("RGB", (640, 384)))
    font = ImageFont.load_default(size=14)
    fitted = renderer._fit_text(draw, "Blue A: controller-name-that-is-far-too-long", 120, font)

    assert fitted.endswith("...")
    assert draw.textlength(fitted, font=font) <= 120


def test_animation_renderer_reports_progress(short_match, tmp_path: Path, capsys) -> None:
    from PIL import Image

    output = render_replay(short_match.replay, tmp_path / "match.gif")
    assert output.stat().st_size > 1_000
    with Image.open(output) as image:
        assert image.size == (640, 384)
        assert image.n_frames == 11
    messages = capsys.readouterr().out
    assert "Encoding" in messages
    assert "Rendering progress: 100%" in messages
    assert "Finished rendering" in messages


def test_mp4_falls_back_to_gif_without_ffmpeg(short_match, tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(renderer.shutil, "which", lambda _name: None)

    output = render_replay(short_match.replay, tmp_path / "match.mp4")

    assert output == tmp_path / "match.gif"
    assert output.stat().st_size > 1_000
    assert "FFmpeg is unavailable; rendering GIF instead." in capsys.readouterr().out


def test_ffmpeg_failure_removes_partial_output(short_match, tmp_path: Path, monkeypatch) -> None:
    class FailedProcess:
        def __init__(self) -> None:
            self.stdin = io.BytesIO()
            self.stderr = io.BytesIO(b"encoder failed")
            self.return_code = None

        def kill(self) -> None:
            self.return_code = -9

        def poll(self):
            return self.return_code

        def wait(self) -> int:
            self.return_code = 1
            return 1

    monkeypatch.setattr(renderer.shutil, "which", lambda _name: "ffmpeg")
    monkeypatch.setattr(renderer.subprocess, "Popen", lambda *_args, **_kwargs: FailedProcess())
    output = tmp_path / "failed.mp4"

    with pytest.raises(RuntimeError, match="FFmpeg failed: encoder failed"):
        render_replay(short_match.replay, output)
    assert not output.exists()


def test_explosion_effect_is_included_in_rendered_frame(short_match, tmp_path: Path) -> None:
    from PIL import Image

    replay = deepcopy(short_match.replay)
    replay.events.append(
        {"time": 0.9, "type": "VEHICLE_COLLISION", "drone_ids": [0, 20], "position": [50.0, 30.0], "team": None, "points": 0}
    )
    assert explosion_events_at(replay, 1.0)
    output = render_replay(replay, tmp_path / "explosion.png")
    assert output.stat().st_size > 1_000
    with Image.open(output).convert("RGB") as image:
        assert any(red > 245 and 130 < green < 210 and blue < 130 for red, green, blue in image.getdata())
    assert not explosion_events_at(replay, 1.3)


def test_renderer_retains_recent_trails(short_match) -> None:
    trails = {}
    for frame in reconstruct_frames(short_match.replay, every_ticks=2):
        renderer._update_trails(frame, short_match.replay, (640, 384), trails)
    assert any(len(points) > 1 for points in trails.values())
    assert all(len(points) <= 20 for points in trails.values())


def test_arena_renderer_creates_image(tmp_path: Path) -> None:
    from PIL import Image

    output = render_arena(generate_scenario(42), tmp_path / "arena.png")
    with Image.open(output) as image:
        assert image.size == (640, 384)


def test_renderer_requires_output_path(short_match) -> None:
    with pytest.raises(ValueError, match="output path is required"):
        render_replay(short_match.replay, None)  # type: ignore[arg-type]


def test_match_metadata_and_result_are_complete(short_match) -> None:
    assert short_match.replay.scenario.seed == 12
    assert len(short_match.replay.controller_a["sha256"]) == 64
    assert short_match.replay.result in {"A", "B", "DRAW"}
    assert short_match.winner in {Team.A, Team.B, None}


def test_match_simulation_is_repeatable_ignoring_wall_clock_metrics(short_match) -> None:
    repeated = run_match(baseline_path("rush"), baseline_path("defend"), seed=12, duration=1.0)
    assert repeated.replay.to_dict() == short_match.replay.to_dict()
