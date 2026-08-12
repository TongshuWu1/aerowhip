from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest
from unittest.mock import patch

from cable_twin.shared.zed_source import (
    CameraCaptureSettings,
    ZedStereoError,
    ZedStereoSource,
    _apply_manual_settings,
    _restore_capture_settings,
    _validate_manual_settings,
)


class _ErrorCode:
    SUCCESS = "success"


class _VideoSettings:
    AEC_AGC = "aec_agc"
    EXPOSURE = "exposure"
    GAIN = "gain"


class _Sdk:
    ERROR_CODE = _ErrorCode
    VIDEO_SETTINGS = _VideoSettings


class _Camera:
    def __init__(self) -> None:
        self.values = {
            _VideoSettings.AEC_AGC: 1,
            _VideoSettings.EXPOSURE: 25,
            _VideoSettings.GAIN: 1,
        }
        self.set_calls: list[tuple[str, int]] = []

    def set_camera_settings(self, setting: str, value: int) -> str:
        self.set_calls.append((setting, int(value)))
        self.values[setting] = int(value)
        if setting in (_VideoSettings.EXPOSURE, _VideoSettings.GAIN):
            self.values[_VideoSettings.AEC_AGC] = 0
        return _ErrorCode.SUCCESS

    def get_camera_settings(self, setting: str) -> tuple[str, int]:
        return _ErrorCode.SUCCESS, self.values[setting]


class ZedCameraSettingsTests(unittest.TestCase):
    def test_manual_values_are_a_complete_integer_pair(self) -> None:
        self.assertEqual(_validate_manual_settings(None, None), (None, None))
        self.assertEqual(_validate_manual_settings(12, 20), (12, 20))
        with self.assertRaises(ValueError):
            _validate_manual_settings(12, None)
        with self.assertRaises(ValueError):
            _validate_manual_settings(-1, 20)
        with self.assertRaises(ValueError):
            _validate_manual_settings(12, 101)
        with self.assertRaises(TypeError):
            _validate_manual_settings(True, 20)

    def test_manual_settings_are_applied_in_safe_order_and_verified(self) -> None:
        camera = _Camera()
        settings = _apply_manual_settings(camera, _Sdk, 12, 20)
        self.assertEqual(
            camera.set_calls,
            [
                (_VideoSettings.EXPOSURE, 12),
                (_VideoSettings.GAIN, 20),
            ],
        )
        self.assertEqual(
            settings,
            CameraCaptureSettings(
                requested_manual=True,
                automatic_exposure_gain=False,
                exposure=12,
                gain=20,
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            settings.exposure = 10  # type: ignore[misc]

    def test_manual_readback_mismatch_is_rejected(self) -> None:
        camera = _Camera()

        def mismatched(setting: str) -> tuple[str, int]:
            value = camera.values[setting]
            if setting == _VideoSettings.EXPOSURE:
                value += 1
            return _ErrorCode.SUCCESS, value

        camera.get_camera_settings = mismatched  # type: ignore[method-assign]
        with self.assertRaisesRegex(ZedStereoError, "did not match"):
            _apply_manual_settings(camera, _Sdk, 12, 20)

    def test_original_automatic_mode_is_restored(self) -> None:
        camera = _Camera()
        camera.values[_VideoSettings.AEC_AGC] = 0
        original = CameraCaptureSettings(
            requested_manual=False,
            automatic_exposure_gain=True,
            exposure=25,
            gain=1,
        )
        _restore_capture_settings(camera, _Sdk, original)
        self.assertEqual(camera.set_calls, [(_VideoSettings.AEC_AGC, 1)])
        self.assertEqual(camera.values[_VideoSettings.AEC_AGC], 1)

    def test_replay_rejects_manual_settings_before_loading_sdk(self) -> None:
        with patch(
            "cable_twin.shared.zed_source._require_zed",
            side_effect=AssertionError("SDK must not be loaded"),
        ):
            with self.assertRaisesRegex(ValueError, "during SVO replay"):
                ZedStereoSource(
                    "recording.svo2",
                    manual_exposure=12,
                    manual_gain=20,
                )


if __name__ == "__main__":
    unittest.main()
