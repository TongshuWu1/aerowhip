from __future__ import annotations

import json

import cv2

from planning.results import latest_planning_result, load_replay_arrays
from planning.video import render_replay_video


def test_saved_replay_is_loaded_without_simulator_execution() -> None:
    result = latest_planning_result("canonical_whip_v1")
    assert result is not None
    arrays = load_replay_arrays(result)
    assert arrays["time_s"].shape == (71,)
    assert arrays["cable_position_m"].shape == (71, 12, 3)
    assert arrays["uav_position_m"].shape == (71, 3)


def test_phone_compatible_video_is_cached_from_authoritative_replay() -> None:
    result = latest_planning_result("canonical_whip_v1")
    assert result is not None
    video, metadata = render_replay_video(result)
    assert video.is_file()
    assert metadata["source_replay_artifact"] == str(result.replay_path)
    assert metadata["render_reran_mppi"] is False
    assert metadata["resolution"] == [1280, 720]
    assert metadata["fps"] == 30
    assert metadata["codec"] == "H.264 (libx264)"
    assert metadata["pixel_format"] == "yuv420p"
    assert metadata["fast_start"] is True
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1280
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 720
    assert capture.get(cv2.CAP_PROP_FPS) == 30.0
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == metadata["frame_count"]
    capture.release()
    saved = json.loads((result.directory / "video_metadata.json").read_text(encoding="utf-8"))
    assert saved["source_trajectory_sha256"] == metadata["source_trajectory_sha256"]

