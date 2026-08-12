from __future__ import annotations

from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest

import numpy as np

from cable_twin.shared.capture_manifest import (
    CAPTURE_MANIFEST_SCHEMA,
    build_capture_manifest,
    load_capture_manifest,
    write_capture_manifest,
)
from cable_twin.shared.contracts import SourceDescriptor, StereoCalibration
from cable_twin.shared.zed_source import CameraCaptureSettings


class CaptureManifestTests(unittest.TestCase):
    @staticmethod
    def _source():
        intrinsics = np.asarray(((800.0, 0.0, 320.0), (0.0, 800.0, 180.0), (0.0, 0.0, 1.0)))
        calibration = StereoCalibration(640, 360, 30.0, intrinsics, intrinsics, 0.12, 1234, "ZED2")
        return SimpleNamespace(
            descriptor=SourceDescriptor("live", "live", calibration),
            capture_settings=CameraCaptureSettings(False, True, 12, 20),
        )

    def test_manifest_records_camera_settings_and_cable_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            svo = Path(temporary) / "motion.svo2"
            svo.write_bytes(b"svo")
            manifest = build_capture_manifest(
                self._source(),
                svo,
                90,
                cable_identity=1,
            )
            self.assertEqual(manifest["schema"], CAPTURE_MANIFEST_SCHEMA)
            self.assertEqual(manifest["frame_count"], 90)
            self.assertTrue(manifest["capture_settings"]["automatic_exposure_gain"])
            self.assertNotIn("release_events", manifest)
            write_capture_manifest(svo, manifest)
            self.assertEqual(load_capture_manifest(svo)["svo_filename"], svo.name)
            self.assertEqual(
                load_capture_manifest(svo)["experiment"]["cable_identity"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
