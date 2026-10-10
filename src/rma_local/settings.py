"""Versioned task settings; old checkpoints always use the legacy task."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskSettings:
    name: str = "legacy"
    train_ranges: tuple = ((0.4, 1.0), (0.0, 2.0), (0.85, 1.15))
    yaw_weight: float = 0.5
    action_delta_weight: float = 0.08
    fall_penalty: float = 2.0
    curriculum_updates: int = 0


def task_settings(name="legacy"):
    if name == "legacy":
        return TaskSettings()
    if name == "robust_v2":
        return TaskSettings(name=name, train_ranges=((0.25, 1.0), (0.0, 3.0), (0.65, 1.15)), yaw_weight=1.0, action_delta_weight=0.16, fall_penalty=5.0, curriculum_updates=1000)
    raise ValueError(f"Unknown task profile: {name}")


def settings_dict(name="legacy"):
    return asdict(task_settings(name))
