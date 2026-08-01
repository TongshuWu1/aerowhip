"""Shared validation for deterministic live-pipeline SVO replay."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .config import RuntimeSettings
from .perception import PerceptionRuntime
from .zed_source import SvoZedSource


def _mapping(root: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = root.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Recording manifest requires {key!r}")
    return value


def validate_replay_manifest(
    manifest: Mapping[str, Any],
    *,
    settings: RuntimeSettings,
    perception: PerceptionRuntime,
    source: SvoZedSource,
) -> None:
    """Reject silent changes to a canonical perception/observation replay."""

    recording = _mapping(manifest, "recording")
    if not bool(recording.get("completed")):
        raise ValueError("Refusing canonical replay of an incomplete recording")
    configuration = _mapping(manifest, "configuration")
    if configuration.get("zed_sdk_version") != source.sdk_version:
        raise ValueError(
            "ZED SDK version differs from the recording: "
            f"recorded={configuration.get('zed_sdk_version')!r}, "
            f"current={source.sdk_version!r}"
        )
    if configuration.get("camera_request") != asdict(settings.camera):
        raise ValueError("Camera/depth configuration differs from the recording")
    recorded_pidnet = _mapping(configuration, "pidnet")
    identities = {
        "runtime_config_sha256": perception.runtime_config_sha256,
        "checkpoint_sha256": perception.checkpoint_sha256,
    }
    for name, current in identities.items():
        if recorded_pidnet.get(name) != current:
            raise ValueError(
                f"PIDNet {name} differs from the recording: "
                f"recorded={recorded_pidnet.get(name)!r}, current={current!r}"
            )
    recorded_observation = configuration.get("observation")
    if recorded_observation is not None:
        current_observation = asdict(settings.observation)
        current_observation["cable_lengths_m"] = list(
            settings.observation.cable_lengths_m
        )
        if recorded_observation != current_observation:
            raise ValueError("Cable observation configuration differs from recording")
    recorded_source = _mapping(manifest, "source")
    recorded_calibration = _mapping(recorded_source, "calibration")
    if dict(recorded_calibration) != asdict(source.descriptor.calibration):
        raise ValueError("SVO calibration does not match its recording manifest")


__all__ = ["validate_replay_manifest"]
