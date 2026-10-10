"""Render fixed GPU evaluation trajectories as videos and tracking plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw


SCENARIOS = ("slow", "fast", "stop_resume", "turn_reversal")
COURSE_SCENARIOS = ('stairs_straight','stairs_stop_resume','slope_straight','slope_turn')
SEEDS = (82000, 82002, 82004, 82006)
EPISODE_INDICES = (0, 2, 4, 6)
FPS = 25
CONTROL_FRAMES_PER_VIDEO_FRAME = 2
WIDTH, HEIGHT = 640, 480


def _load_data(evaluation_dir: Path):
    summary_path = evaluation_dir / "evaluation.json"
    trajectory_path = evaluation_dir / "trajectories.npz"
    if not summary_path.is_file() or not trajectory_path.is_file():
        raise FileNotFoundError("evaluation must contain evaluation.json and trajectories.npz")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = summary.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 8:
        raise ValueError("evaluation.json must contain eight episodes in the prescribed order")
    data = np.load(trajectory_path)
    required = {"qpos", "qvel", "velocity", "command", "valid", "action"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"trajectories.npz is missing keys: {', '.join(sorted(missing))}")
    expected = {
        "qpos": (8, 19), "qvel": (8, 18), "velocity": (8, 3),
        "command": (8, 3), "valid": (8,), "action": (8, 12),
    }
    for name, tail in expected.items():
        if data[name].ndim != 1 + len(tail) or data[name].shape[1:] != tail:
            raise ValueError(f"{name} must have shape [T,{','.join(map(str, tail))}]")
    n = data["qpos"].shape[0]
    if n == 0 or any(data[name].shape[0] != n for name in required):
        raise ValueError("trajectory arrays must have the same nonzero frame count")
    scenarios=COURSE_SCENARIOS if summary.get('terrain')=='a1_course' else SCENARIOS
    for i, scenario in enumerate(scenarios):
        row = episodes[EPISODE_INDICES[i]]
        if row.get("scenario") != scenario or not isinstance(row.get("seed"),int):
            raise ValueError(f"episode {EPISODE_INDICES[i]} must be {scenario} with an integer seed")
        steps = row.get("steps")
        if not isinstance(steps, int) or not 0 < steps <= n:
            raise ValueError(f"invalid steps for episode {EPISODE_INDICES[i]}")
    return data, episodes


def _render_episode(renderer, model, data, arrays, summary_row, episode_index, scenario, video_path):
    valid = np.asarray(arrays["valid"][:summary_row["steps"], episode_index], dtype=bool)
    qpos = arrays["qpos"][:summary_row["steps"], episode_index]
    if not valid.any():
        raise ValueError(f"episode {episode_index} has no valid frames")
    # Stop at the first invalid step; the evaluator records post-fall states but
    # those must never appear as a revived or continued robot in the video.
    first_invalid = np.flatnonzero(~valid)
    end = int(first_invalid[0]) if len(first_invalid) else len(valid)
    end = max(1, end)
    with imageio.get_writer(video_path, fps=FPS, codec="libx264") as writer:
        for frame in range(0, end, CONTROL_FRAMES_PER_VIDEO_FRAME):
            data.qpos[:] = arrays["qpos"][frame, episode_index]
            data.qvel[:] = arrays["qvel"][frame, episode_index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="track")
            rgb = renderer.render()
            cmd = arrays["command"][frame, episode_index]
            t = frame * 0.02
            title = f"GPU trained A1 / {scenario} / t={t:.2f}s / cmd vx={cmd[0]:.2f} yaw={cmd[2]:.2f}"
            # ASCII-only annotation for portability across system fonts.
            canvas = Image.fromarray(np.asarray(rgb))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 8), title, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
            if frame + CONTROL_FRAMES_PER_VIDEO_FRAME >= end:
                status = "PASSED" if summary_row.get("tracking_success") else "FAILED"
                draw.text((10, HEIGHT - 28), status,
                          fill=(255, 80, 80) if status == "FAILED" else (80, 255, 80),
                          stroke_width=1, stroke_fill=(0, 0, 0))
            writer.append_data(np.asarray(canvas))


def _plot_tracking(arrays, episodes, out_path):
    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex="row")
    colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
    scenarios=tuple(episodes[i]['scenario'] for i in EPISODE_INDICES)
    for row, (scenario, episode_index, color) in enumerate(zip(scenarios, EPISODE_INDICES, colors)):
        episode = episodes[episode_index]
        n = episode["steps"]
        mask = np.asarray(arrays["valid"][:n, episode_index], dtype=bool)
        ids = np.flatnonzero(mask)
        t = ids / FPS / CONTROL_FRAMES_PER_VIDEO_FRAME
        for col, (component, label) in enumerate(((0, "vx"), (2, "yaw rate"))):
            ax = axes[row, col]
            ax.plot(t, arrays["command"][ids, episode_index, component], "--", color="black", label="command")
            ax.plot(t, arrays["velocity"][ids, episode_index, component], color=color, label="actual")
            ax.set_ylabel(f"{scenario}\n{label}")
            ax.grid(True, alpha=0.25)
            if row == 0:
                ax.legend(loc="best")
            if row == 3:
                ax.set_xlabel("Time (s)")
    fig.suptitle("GPU trained A1: commanded and actual velocity (valid frames only)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def render(assets_dir: Path, evaluation_dir: Path, terrain: str, out_dir: Path, episode_indices=EPISODE_INDICES):
    xml_path = assets_dir / f"a1_{terrain}.xml"
    asset_dir = assets_dir / "assets"
    if not xml_path.is_file() or not asset_dir.is_dir():
        raise FileNotFoundError(f"assets must contain {xml_path.name} and assets/")
    arrays, episodes = _load_data(evaluation_dir)
    # Parse XML with the original MuJoCo Python package, loading sibling assets.
    model_assets = {p.name: p.read_bytes() for p in asset_dir.iterdir() if p.is_file()}
    model = mujoco.MjModel.from_xml_string(xml_path.read_text(encoding="utf-8"), assets=model_assets)
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    model.opt.timestep = 0.02
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    data = mujoco.MjData(model)
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        for episode_index in episode_indices:
            scenario=episodes[episode_index]['scenario']
            _render_episode(renderer, model, data, arrays, episodes[episode_index], episode_index, scenario,
                            out_dir / f"{scenario}_seed{episodes[episode_index]['seed']}.mp4")
    _plot_tracking(arrays, episodes, out_dir / "tracking.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path, help="directory containing a1_flat.xml/a1_rough.xml and assets/")
    parser.add_argument("--evaluation", required=True, type=Path, help="directory containing evaluation.json and trajectories.npz")
    parser.add_argument("--terrain", choices=("flat", "rough", "course"), required=True)
    parser.add_argument("--out", required=True, type=Path, help="new output directory; existing directories are rejected")
    parser.add_argument("--episodes",type=int,nargs='+',default=list(EPISODE_INDICES),choices=range(8))
    args = parser.parse_args()
    render(args.assets, args.evaluation, args.terrain, args.out,args.episodes)


if __name__ == "__main__":
    main()
