"""Fast Pillow replay rendering kept separate from authoritative simulation."""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from swarmbench.api import PHYSICS_DT, CircleObstacle, DroneStatus, DroneType, RectangleObstacle, Team

from .format import Replay, ReplayFrame, reconstruct_frames


@dataclass(frozen=True, slots=True)
class _KillfeedEntry:
    time: float
    kind: str
    vehicles: tuple[tuple[Team, DroneType], ...]
    points: int = 0


def explosion_events_at(replay: Replay, timestamp: float, lifetime: float = 0.25) -> list[dict[str, Any]]:
    return [
        event
        for event in replay.events
        if event.get("type") in {"VEHICLE_COLLISION", "PROJECTILE_HIT", "OBSTACLE_CRASH"}
        and 0.0 <= timestamp - float(event["time"]) <= lifetime
    ]


def _killfeed_entries(replay: Replay) -> tuple[_KillfeedEntry, ...]:
    drones = {drone.id: drone for drone in replay.scenario.drones}
    projectile_sources: dict[int, int] = {}
    entries: list[_KillfeedEntry] = []
    for event in replay.events:
        event_type = event.get("type")
        drone_ids = event.get("drone_ids", [])
        projectile_id = event.get("projectile_id")
        if event_type == "PROJECTILE_FIRED":
            if projectile_id is not None and drone_ids:
                projectile_sources[int(projectile_id)] = int(drone_ids[0])
            continue

        vehicles = []
        for drone_id in drone_ids:
            drone = drones.get(int(drone_id))
            if drone is not None:
                vehicles.append((drone.team, drone.drone_type))
        if event_type == "PROJECTILE_HIT" and vehicles:
            source_id = projectile_sources.get(int(projectile_id)) if projectile_id is not None else None
            source = drones.get(source_id) if source_id is not None else None
            if source is not None:
                attacker = (source.team, source.drone_type)
            elif event.get("team") in {Team.A.value, Team.B.value}:
                attacker = (Team(event["team"]), DroneType.TANK)
            else:
                continue
            entries.append(_KillfeedEntry(float(event["time"]), "hit", (attacker, vehicles[0])))
        elif event_type == "VEHICLE_COLLISION" and len(vehicles) == 2:
            entries.append(_KillfeedEntry(float(event["time"]), "collision", tuple(vehicles)))
        elif event_type == "OBSTACLE_CRASH" and vehicles:
            entries.append(_KillfeedEntry(float(event["time"]), "crash", (vehicles[0],)))
        elif event_type == "GOAL" and vehicles:
            entries.append(_KillfeedEntry(float(event["time"]), "score", (vehicles[0],), int(event.get("points", 0))))
    return tuple(sorted(entries, key=lambda entry: entry.time))


def _visible_killfeed_entries(
    entries: tuple[_KillfeedEntry, ...],
    timestamp: float,
    *,
    duration: float,
    limit: int,
) -> tuple[_KillfeedEntry, ...]:
    visible = [entry for entry in entries if 0.0 <= timestamp - entry.time <= duration]
    return tuple(reversed(visible[-limit:]))


def _quality_settings(quality: str) -> tuple[tuple[int, int], int, int, int, int]:
    if quality == "low":
        return (640, 384), 64, 1, 75, 1000
    if quality == "high":
        return (1400, 840), 256, 6, 92, 1800
    raise ValueError("quality must be 'low' or 'high'")


def _point(position: tuple[float, float] | list[float], replay: Replay, size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    return (
        round(float(position[0]) * (width - 1) / replay.scenario.width),
        round((replay.scenario.height - float(position[1])) * (height - 1) / replay.scenario.height),
    )


def _rectangle(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    replay: Replay,
    size: tuple[int, int],
) -> tuple[int, int, int, int]:
    left, bottom = _point((x_min, y_min), replay, size)
    right, top = _point((x_max, y_max), replay, size)
    return left, top, right, bottom


def _arena_background(replay: Replay, size: tuple[int, int]) -> Any:
    from PIL import Image, ImageDraw

    width, height = size
    scale = width / 640
    image = Image.new("RGB", size, "#f7f7f2")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, width - 1, height - 1), outline="#222222", width=max(1, round(2 * scale)))
    for goal, color in ((replay.scenario.goal_for_a, "#4f8dd6"), (replay.scenario.goal_for_b, "#dc5a5a")):
        draw.rectangle(
            _rectangle(goal.x_min, goal.x_max, goal.y_min, goal.y_max, replay, size),
            fill=color + "33",
            outline=color,
            width=max(1, round(scale)),
        )
    for obstacle in replay.scenario.obstacles:
        if isinstance(obstacle, CircleObstacle):
            x, y = _point(obstacle.center, replay, size)
            radius = round(obstacle.radius * (width - 1) / replay.scenario.width)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#555555")
        elif isinstance(obstacle, RectangleObstacle):
            draw.rectangle(
                _rectangle(obstacle.x_min, obstacle.x_max, obstacle.y_min, obstacle.y_max, replay, size),
                fill="#555555",
            )
    return image


def _update_trails(
    frame: ReplayFrame,
    replay: Replay,
    size: tuple[int, int],
    trails: dict[int, list[tuple[int, int]]],
) -> None:
    for drone in frame.drones:
        if drone.status is not DroneStatus.ACTIVE:
            continue
        trail = trails.setdefault(drone.id, [])
        trail.append(_point(drone.position, replay, size))
        del trail[:-20]


def _star(center: tuple[int, int], radius: float) -> list[tuple[float, float]]:
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        length = radius if index % 2 == 0 else radius * 0.45
        points.append((center[0] + math.cos(angle) * length, center[1] + math.sin(angle) * length))
    return points


def _remaining_counts(frame: ReplayFrame, team: Team) -> tuple[int, int, int]:
    return tuple(
        sum(
            drone.status is DroneStatus.ACTIVE and drone.team is team and drone.drone_type is drone_type
            for drone in frame.drones
        )
        for drone_type in (DroneType.SCOUT, DroneType.TRANSPORT, DroneType.TANK)
    )


def _count_text(counts: tuple[int, int, int]) -> str:
    scout, transport, tank = counts
    return f"S {scout}  Tr {transport}  Tk {tank}"


def _fit_text(draw: Any, value: str, max_width: int, font: Any) -> str:
    if draw.textlength(value, font=font) <= max_width:
        return value
    suffix = "..."
    while value and draw.textlength(value + suffix, font=font) > max_width:
        value = value[:-1]
    return value + suffix


def _draw_vehicle_icon(
    draw: Any,
    center: tuple[float, float],
    vehicle: tuple[Team, DroneType],
    radius: int,
) -> None:
    team, drone_type = vehicle
    color = "#1769aa" if team is Team.A else "#c62828"
    x, y = center
    bounds = (x - radius, y - radius, x + radius, y + radius)
    if drone_type is DroneType.SCOUT:
        draw.ellipse(bounds, fill=color)
    elif drone_type is DroneType.TRANSPORT:
        draw.rectangle(bounds, fill=color)
    else:
        draw.polygon(((x, y - radius), (x - radius, y + radius), (x + radius, y + radius)), fill=color)


def _draw_killfeed(
    draw: Any,
    entries: tuple[_KillfeedEntry, ...],
    size: tuple[int, int],
    font: Any,
) -> None:
    if not entries:
        return
    width, height = size
    scale = width / 640
    edge_gap = max(4, round(5 * scale))
    padding = max(4, round(5 * scale))
    row_height = max(18, round(20 * scale))
    panel_width = max(118, round(135 * scale))
    panel_left = (width - panel_width) // 2
    panel_right = panel_left + panel_width
    panel_bottom = height - edge_gap
    panel_top = panel_bottom - padding - len(entries) * row_height
    draw.rounded_rectangle(
        (panel_left, panel_top, panel_right, panel_bottom),
        radius=max(3, round(4 * scale)),
        fill=(247, 247, 242, 130),
        outline=(34, 34, 34, 65),
        width=max(1, round(scale)),
    )
    icon_radius = max(4, round(4 * scale))
    icon_x = panel_left + padding + icon_radius
    for index, entry in enumerate(entries):
        center_y = panel_top + padding // 2 + row_height * index + row_height / 2
        _draw_vehicle_icon(draw, (icon_x, center_y), entry.vehicles[0], icon_radius)
        if entry.kind in {"hit", "collision"}:
            arrow = "->" if entry.kind == "hit" else "<->"
            arrow_x = icon_x + round(20 * scale)
            draw.text((arrow_x, center_y), arrow, fill=(34, 34, 34, 220), font=font, anchor="mm")
            _draw_vehicle_icon(draw, (icon_x + round(40 * scale), center_y), entry.vehicles[1], icon_radius)
        else:
            label = "crash" if entry.kind == "crash" else f"score (+{entry.points})"
            draw.text(
                (icon_x + round(13 * scale), center_y),
                label,
                fill=(34, 34, 34, 220),
                font=font,
                anchor="lm",
            )


def _draw_frame(
    background: Any,
    replay: Replay,
    frame: ReplayFrame,
    trails: dict[int, list[tuple[int, int]]],
    size: tuple[int, int],
    font: Any,
    killfeed_entries: tuple[_KillfeedEntry, ...] = (),
    killfeed_duration: float = 5.0,
    killfeed_lines: int = 5,
    killfeed_font: Any | None = None,
) -> Any:
    from PIL import ImageDraw

    width, _ = size
    scale = width / 640
    image = background.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    for drone in frame.drones:
        if drone.status is not DroneStatus.ACTIVE:
            continue
        color = "#1769aa" if drone.team is Team.A else "#c62828"
        trail = trails.get(drone.id, [])
        if len(trail) > 1:
            draw.line(trail, fill=color + "2e", width=max(1, round(scale)))
        x, y = _point(drone.position, replay, size)
        radius = max(2, round((3 if drone.drone_type is DroneType.SCOUT else 4) * scale))
        bounds = (x - radius, y - radius, x + radius, y + radius)
        if drone.drone_type is DroneType.SCOUT:
            draw.ellipse(bounds, fill=color)
        elif drone.drone_type is DroneType.TRANSPORT:
            draw.rectangle(bounds, fill=color)
        else:
            draw.polygon(((x, y - radius - 1), (x - radius, y + radius), (x + radius, y + radius)), fill=color)
    for projectile in frame.projectiles:
        color = "#64b5f6" if projectile.team is Team.A else "#ef5350"
        x, y = _point(projectile.position, replay, size)
        radius = max(2, round(2 * scale))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="#ffffff")
    for event in explosion_events_at(replay, frame.time):
        age = frame.time - float(event["time"])
        radius = (4 + 8 * age / 0.25) * scale
        alpha = round(255 * max(0.1, 1 - age / 0.25))
        draw.polygon(_star(_point(event["position"], replay, size), radius), fill=(255, 143, 0, alpha))
    banner_height = max(40, round(44 * scale))
    draw.rectangle((0, 0, width, banner_height), fill=(247, 247, 242, 225))
    padding = max(6, round(8 * scale))
    name_y = max(1, round(2 * scale))
    details_y = max(18, round(21 * scale))
    name_width = width // 2 - 2 * padding
    controller_a = replay.controller_a.get("id", "unknown") or "unknown"
    controller_b = replay.controller_b.get("id", "unknown") or "unknown"
    left_name = _fit_text(draw, f"Blue A: {controller_a}", name_width, font)
    right_name = _fit_text(draw, f"Red B: {controller_b}", name_width, font)
    draw.text((padding, name_y), left_name, fill="#1769aa", font=font)
    draw.text(
        (width - padding - round(draw.textlength(right_name, font=font)), name_y),
        right_name,
        fill="#c62828",
        font=font,
    )

    left_counts = _count_text(_remaining_counts(frame, Team.A))
    status = f"t={frame.time:05.2f}s   {frame.scores[0]} - {frame.scores[1]}"
    right_counts = _count_text(_remaining_counts(frame, Team.B))
    draw.text((padding, details_y), left_counts, fill="#1769aa", font=font)
    draw.text(
        ((width - draw.textlength(status, font=font)) / 2, details_y),
        status,
        fill="#222222",
        font=font,
    )
    draw.text(
        (width - padding - round(draw.textlength(right_counts, font=font)), details_y),
        right_counts,
        fill="#c62828",
        font=font,
    )
    _draw_killfeed(
        draw,
        _visible_killfeed_entries(
            killfeed_entries,
            frame.time,
            duration=killfeed_duration,
            limit=killfeed_lines,
        ),
        size,
        killfeed_font or font,
    )
    return image


def _frame_count(replay: Replay, every_ticks: int) -> int:
    total_ticks = round(replay.final_time / PHYSICS_DT)
    return 1 + total_ticks // every_ticks + int(total_ticks % every_ticks != 0)


def _write_gif(
    destination: Path,
    images: Iterable[Any],
    *,
    fps: int,
    palette_colors: int,
    report_progress: Callable[[int], None],
) -> None:
    from PIL import GifImagePlugin, Image

    try:
        with destination.open("wb") as stream:
            count = 0
            for index, image in enumerate(images):
                frame = image.convert(
                    "P",
                    palette=Image.Palette.ADAPTIVE,
                    colors=palette_colors,
                    dither=Image.Dither.NONE,
                )
                if index == 0:
                    header, _ = GifImagePlugin.getheader(frame, info={"loop": 0})
                    for chunk in header:
                        stream.write(chunk)
                for chunk in GifImagePlugin.getdata(
                    frame,
                    duration=round(1000 / fps),
                    disposal=2,
                    include_color_table=index > 0,
                ):
                    stream.write(chunk)
                count += 1
                report_progress(index)
            if count == 0:
                raise RuntimeError("replay produced no frames")
            stream.write(b";")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _write_mp4(
    destination: Path,
    images: Iterable[Any],
    *,
    ffmpeg: str,
    fps: int,
    size: tuple[int, int],
    bitrate: int,
    report_progress: Callable[[int], None],
) -> None:
    command = [
        ffmpeg,
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{size[0]}x{size[1]}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        f"{bitrate}k",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if process.stdin is None or process.stderr is None:
        process.kill()
        destination.unlink(missing_ok=True)
        raise RuntimeError("failed to open FFmpeg pipes")
    try:
        for index, image in enumerate(images):
            process.stdin.write(image.tobytes())
            report_progress(index)
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace").strip()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg failed: {error or f'exit code {return_code}'}")
    except (BrokenPipeError, OSError) as write_error:
        if not process.stdin.closed:
            try:
                process.stdin.close()
            except OSError:
                pass
        error = process.stderr.read().decode("utf-8", errors="replace").strip()
        process.wait()
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg failed: {error or write_error}") from write_error
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        destination.unlink(missing_ok=True)
        raise


def render_replay(
    replay: Replay,
    output: str | Path,
    *,
    fps: int = 10,
    quality: str = "low",
    killfeed: bool = False,
    killfeed_duration: float = 5.0,
    killfeed_lines: int = 5,
) -> Path:
    from PIL import ImageFont

    if output is None:
        raise ValueError("an output path is required")
    source_fps = round(1 / PHYSICS_DT)
    if fps not in {5, 10, source_fps}:
        raise ValueError(f"fps must be one of 5, 10, or {source_fps}")
    if killfeed and killfeed_duration <= 0:
        raise ValueError("killfeed duration must be positive")
    if killfeed and killfeed_lines <= 0:
        raise ValueError("killfeed lines must be positive")
    size, palette_colors, png_compression, jpeg_quality, bitrate = _quality_settings(quality)
    every_ticks = source_fps // fps
    total_frames = _frame_count(replay, every_ticks)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    print("Reconstructing replay frames...", flush=True)
    print(f"Prepared {total_frames} frames at {fps} FPS ({quality} quality).", flush=True)
    background = _arena_background(replay, size)
    font = ImageFont.load_default(size=max(12, round(14 * size[0] / 640)))
    killfeed_font = ImageFont.load_default(size=max(9, round(10 * size[0] / 640)))
    trails: dict[int, list[tuple[int, int]]] = {}
    killfeed_entries = _killfeed_entries(replay) if killfeed else ()

    if destination.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        final_frame = None
        for final_frame in reconstruct_frames(replay, every_ticks=every_ticks):
            _update_trails(final_frame, replay, size, trails)
        if final_frame is None:
            raise RuntimeError("replay produced no frames")
        print(f"Rendering final frame to {destination}...", flush=True)
        image = _draw_frame(
            background,
            replay,
            final_frame,
            trails,
            size,
            font,
            killfeed_entries,
            killfeed_duration,
            killfeed_lines,
            killfeed_font,
        )
        if destination.suffix.lower() == ".png":
            image.save(destination, compress_level=png_compression)
        else:
            image.save(destination, quality=jpeg_quality)
        print(f"Finished rendering {destination}.", flush=True)
        return destination

    def images() -> Iterable[Any]:
        for frame in reconstruct_frames(replay, every_ticks=every_ticks):
            _update_trails(frame, replay, size, trails)
            yield _draw_frame(
                background,
                replay,
                frame,
                trails,
                size,
                font,
                killfeed_entries,
                killfeed_duration,
                killfeed_lines,
                killfeed_font,
            )

    last_reported = -1

    def report_progress(index: int) -> None:
        nonlocal last_reported
        percent = min(100, (index + 1) * 100 // total_frames)
        milestone = percent // 10 * 10
        if milestone > last_reported:
            last_reported = milestone
            print(f"Rendering progress: {milestone}%", flush=True)

    ffmpeg = shutil.which("ffmpeg")
    if destination.suffix.lower() == ".mp4" and ffmpeg is not None:
        print(f"Encoding {total_frames} frames to {destination}...", flush=True)
        _write_mp4(
            destination,
            images(),
            ffmpeg=ffmpeg,
            fps=fps,
            size=size,
            bitrate=bitrate,
            report_progress=report_progress,
        )
    else:
        if destination.suffix.lower() != ".gif":
            if destination.suffix.lower() == ".mp4":
                print("FFmpeg is unavailable; rendering GIF instead.", flush=True)
            destination = destination.with_suffix(".gif")
        print(f"Encoding {total_frames} frames to {destination}...", flush=True)
        _write_gif(
            destination,
            images(),
            fps=fps,
            palette_colors=palette_colors,
            report_progress=report_progress,
        )
    print(f"Finished rendering {destination}.", flush=True)
    return destination


def render_arena(scenario: Any, output: str | Path) -> Path:
    replay = Replay(
        scenario,
        {"id": "none", "sha256": ""},
        {"id": "none", "sha256": ""},
        [],
        [],
        0.0,
        {"A": 0, "B": 0},
        "DRAW",
    )
    return render_replay(replay, output)
