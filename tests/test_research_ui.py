from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from research_ui import catalog
from research_ui import app


class ResearchCatalogTests(unittest.TestCase):
    def test_workflow_catalog_is_ordered_unique_and_immutable(self) -> None:
        self.assertEqual(len(catalog.WORKFLOWS), 3)
        identifiers = [workflow.workflow_id for workflow in catalog.WORKFLOWS]
        self.assertEqual(identifiers, ["identify", "policy", "online"])
        self.assertEqual(
            [workflow.stage for workflow in catalog.WORKFLOWS],
            [1, 2, 3],
        )
        self.assertTrue(all(workflow.primary for workflow in catalog.WORKFLOWS))

        with self.assertRaises(FrozenInstanceError):
            catalog.WORKFLOWS[0].title = "changed"  # type: ignore[misc]

    def test_root_exposes_exactly_three_public_launchers(self) -> None:
        launchers = {
            path.name for path in catalog.REPOSITORY_DIRECTORY.glob("run*.py")
        }
        self.assertEqual(
            launchers,
            {
                "run_offline_fitting.py",
                "run_sac_training.py",
                "run_online.py",
            },
        )

    def test_launch_specs_use_absolute_list_form_commands(self) -> None:
        expected_python = str(Path(sys.executable).resolve())
        expected_cwd = catalog.REPOSITORY_DIRECTORY.resolve()
        for workflow in catalog.WORKFLOWS:
            with self.subTest(workflow=workflow.workflow_id):
                spec = catalog.build_launch_spec(workflow.workflow_id)
                self.assertIsInstance(spec.command, tuple)
                self.assertEqual(spec.command[0], expected_python)
                self.assertEqual(
                    spec.command[1],
                    str((expected_cwd / workflow.script_name).resolve()),
                )
                self.assertEqual(len(spec.command), 2)
                self.assertEqual(spec.cwd, expected_cwd)
                self.assertTrue(Path(spec.command[1]).is_file())

        with self.assertRaisesRegex(KeyError, "Unknown research workflow"):
            catalog.build_launch_spec("not-a-workflow")

    def test_catalog_import_does_not_load_tk_or_compute_stacks(self) -> None:
        statement = (
            "import sys; import research_ui.catalog; "
            "blocked=('tkinter','torch','cv2','pyzed'); "
            "loaded=[name for name in blocked if name in sys.modules]; "
            "raise SystemExit('unexpected imports: '+','.join(loaded) if loaded else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", statement],
            cwd=catalog.REPOSITORY_DIRECTORY,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_artifact_inspection_reports_provenance_without_loading_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_directory = root / "csv"
            csv_directory.mkdir()
            (csv_directory / "take.csv").write_text("Frame,c1 X\n1,0.0\n", encoding="utf-8")

            model = root / "model.json"
            model.write_text(
                json.dumps({"schema": "test_cable_model_v1", "optimized": {"EI": 1.0}}),
                encoding="utf-8",
            )
            policy = root / "policy.pt"
            policy.write_bytes(b"policy-checkpoint")
            policy.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "schema": "test_sac_policy_v1",
                        "source_model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )

            trial_directory = root / "trials"
            trial_directory.mkdir()
            (trial_directory / "trial.npz").write_bytes(b"trial")

            artifact_paths = {
                "optitrack_csv": ("OptiTrack CSV collection", csv_directory),
                "cable_model": ("Nominal cable model", model),
                "sac_policy": ("Selected SAC policy", policy),
                "adaptation_trials": ("Adaptation trials", trial_directory),
            }
            with (
                mock.patch.object(catalog, "REPOSITORY_DIRECTORY", root),
                mock.patch.object(catalog, "_ARTIFACT_PATHS", artifact_paths),
            ):
                statuses = catalog.inspect_artifacts()

            by_id = {status.artifact_id: status for status in statuses}
            self.assertEqual(
                [status.artifact_id for status in statuses],
                [
                    "optitrack_csv",
                    "cable_model",
                    "sac_policy",
                    "adaptation_trials",
                ],
            )
            self.assertEqual(by_id["optitrack_csv"].item_count, 1)
            self.assertEqual(by_id["cable_model"].schema, "test_cable_model_v1")
            self.assertEqual(
                by_id["cable_model"].sha256,
                hashlib.sha256(model.read_bytes()).hexdigest(),
            )
            self.assertEqual(by_id["sac_policy"].kind, "file")
            self.assertEqual(by_id["sac_policy"].schema, "test_sac_policy_v1")
            self.assertEqual(by_id["sac_policy"].compatibility, "compatible")
            self.assertEqual(by_id["sac_policy"].status_label, "COMPATIBLE")
            self.assertTrue(all(status.exists for status in statuses))
            self.assertTrue(all(status.sha256 for status in statuses))

    def test_policy_sidecar_mismatch_is_reported_without_loading_torch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.json"
            model.write_text(
                json.dumps({"schema": "test_cable_model_v1"}),
                encoding="utf-8",
            )
            policy = root / "policy.pt"
            policy.write_bytes(b"policy-checkpoint")
            policy.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "schema": "test_sac_policy_v1",
                        "source_model_sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            artifact_paths = {
                "optitrack_csv": ("OptiTrack CSV collection", root / "csv"),
                "cable_model": ("Nominal cable model", model),
                "sac_policy": ("Selected SAC policy", policy),
                "adaptation_trials": ("Adaptation trials", root / "trials"),
            }
            with (
                mock.patch.object(catalog, "REPOSITORY_DIRECTORY", root),
                mock.patch.object(catalog, "_ARTIFACT_PATHS", artifact_paths),
            ):
                statuses = catalog.inspect_artifacts()
            policy_status = next(
                status for status in statuses if status.artifact_id == "sac_policy"
            )
            self.assertEqual(policy_status.compatibility, "mismatch")
            self.assertEqual(policy_status.status_label, "MISMATCH")

    def test_artifact_status_never_equates_existence_with_scientific_readiness(self) -> None:
        common = {
            "label": "artifact",
            "path": Path("artifact"),
            "kind": "file",
            "item_count": 1,
            "size_bytes": 1,
            "modified_text": "now",
            "sha256": "a" * 64,
        }
        provisional = catalog.ArtifactStatus(
            artifact_id="cable_model",
            exists=True,
            schema="optitrack_twist_aware_rod_v5",
            **common,
        )
        available = catalog.ArtifactStatus(
            artifact_id="sac_policy",
            exists=True,
            schema=None,
            **common,
        )
        missing = catalog.ArtifactStatus(
            artifact_id="sac_policy",
            exists=False,
            schema=None,
            **common,
        )
        self.assertEqual(provisional.status_label, "PROVISIONAL")
        self.assertEqual(available.status_label, "AVAILABLE")
        self.assertEqual(missing.status_label, "MISSING")

    def test_missing_and_invalid_artifacts_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid_model = root / "model.json"
            invalid_model.write_text("not JSON", encoding="utf-8")
            artifact_paths = {
                "optitrack_csv": ("OptiTrack CSV collection", root / "missing_csv"),
                "cable_model": ("Nominal cable model", invalid_model),
                "sac_policy": ("Selected SAC policy", root / "missing.pt"),
                "adaptation_trials": ("Adaptation trials", root / "missing_trials"),
            }
            with (
                mock.patch.object(catalog, "REPOSITORY_DIRECTORY", root),
                mock.patch.object(catalog, "_ARTIFACT_PATHS", artifact_paths),
            ):
                statuses = catalog.inspect_artifacts()

            by_id = {status.artifact_id: status for status in statuses}
            self.assertEqual(by_id["cable_model"].schema, "invalid JSON")
            self.assertTrue(by_id["cable_model"].exists)
            for artifact_id in (
                "optitrack_csv",
                "sac_policy",
                "adaptation_trials",
            ):
                self.assertFalse(by_id[artifact_id].exists)
                self.assertIsNone(by_id[artifact_id].sha256)

    def test_collection_digest_includes_stable_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_directory = root / "library_a"
            second_directory = root / "library_b"
            first_directory.mkdir()
            second_directory.mkdir()
            first = first_directory / "demo.npz"
            second = second_directory / "demo.npz"
            first.write_bytes(b"same-demonstration")
            second.write_bytes(b"same-demonstration")
            with mock.patch.object(catalog, "REPOSITORY_DIRECTORY", root):
                first_digest = catalog._directory_digest((first,))
                second_digest = catalog._directory_digest((second,))
            self.assertNotEqual(first_digest, second_digest)


class ResearchLauncherProcessTests(unittest.TestCase):
    def test_app_import_does_not_load_cuda_or_domain_backends(self) -> None:
        statement = (
            "import sys; import research_ui.app; "
            "blocked=('torch','cv2','pyzed','drone_mpc.adaptation_testbed'); "
            "loaded=[name for name in blocked if name in sys.modules]; "
            "raise SystemExit('unexpected imports: '+','.join(loaded) if loaded else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", statement],
            cwd=catalog.REPOSITORY_DIRECTORY,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_launch_uses_child_process_without_shell_or_captured_pipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "tool.py"
            spec = catalog.LaunchSpec(
                workflow_id="test",
                command=(str(Path(sys.executable).resolve()), str(script)),
                cwd=root,
            )
            process = mock.Mock(pid=1234)
            with mock.patch.object(app.subprocess, "Popen", return_value=process) as popen:
                launched = app.launch_workflow_process(spec, root / "logs" / "run.log")
            try:
                popen.assert_called_once()
                args, kwargs = popen.call_args
                self.assertEqual(args, (list(spec.command),))
                self.assertEqual(kwargs["cwd"], str(root))
                self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
                self.assertIs(kwargs["stderr"], subprocess.STDOUT)
                self.assertIs(kwargs["stdout"], launched.log_stream)
                self.assertFalse(kwargs["shell"])
                self.assertNotIn("start_new_session", kwargs)
                self.assertEqual(launched.process, process)
                self.assertEqual(launched.log_path, (root / "logs" / "run.log").resolve())
                self.assertTrue(launched.log_path.is_file())
            finally:
                launched.log_stream.close()

    def test_launch_log_starts_with_machine_readable_session_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = catalog.LaunchSpec(
                workflow_id="online",
                command=(str(Path(sys.executable).resolve()), str(root / "tool.py")),
                cwd=root,
            )
            manifest = {
                "workflow_id": "online",
                "command": list(spec.command),
                "cwd": str(root),
                "artifacts_at_launch": {
                    "cable_model": {
                        "sha256": "a" * 64,
                        "schema": "model_v1",
                        "status": "AVAILABLE",
                    }
                },
            }
            process = mock.Mock(pid=1234)
            with mock.patch.object(app.subprocess, "Popen", return_value=process):
                launched = app.launch_workflow_process(
                    spec,
                    root / "logs" / "run.log",
                    manifest=manifest,
                )
            launched.log_stream.close()
            content = launched.log_path.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("# CABLE WHIP RESEARCH SESSION\n"))
            self.assertIn('"workflow_id": "online"', content)
            self.assertIn('"command": [', content)
            self.assertIn('"sha256": "' + "a" * 64 + '"', content)
            self.assertTrue(content.endswith("# CHILD PROCESS OUTPUT\n"))

    def test_launch_failure_closes_the_session_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = catalog.LaunchSpec(
                workflow_id="test",
                command=(str(Path(sys.executable).resolve()), str(root / "tool.py")),
                cwd=root,
            )
            log_path = root / "logs" / "failed.log"
            with (
                mock.patch.object(app.subprocess, "Popen", side_effect=OSError("failed")),
                self.assertRaisesRegex(OSError, "failed"),
            ):
                app.launch_workflow_process(spec, log_path)
            # On Windows this unlink would fail if the error path leaked its
            # file handle.  It also leaves the test directory deterministic.
            log_path.unlink()

    def test_closing_console_never_terminates_child_research_tools(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        stream = mock.Mock(closed=False)
        root = mock.Mock()
        console = SimpleNamespace(
            running={
                "online": app.RunningWorkflow(
                    process=process,
                    log_stream=stream,
                    log_path=Path("trial.log"),
                )
            },
            root=root,
        )
        with mock.patch.object(app.messagebox, "askyesno", return_value=True):
            app.ResearchConsole.close(console)  # type: ignore[arg-type]
        stream.close.assert_called_once_with()
        root.destroy.assert_called_once_with()
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_user_can_cancel_console_close_while_child_is_running(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        stream = mock.Mock(closed=False)
        root = mock.Mock()
        console = SimpleNamespace(
            running={
                "online": app.RunningWorkflow(
                    process=process,
                    log_stream=stream,
                    log_path=Path("trial.log"),
                )
            },
            root=root,
        )
        with mock.patch.object(app.messagebox, "askyesno", return_value=False):
            app.ResearchConsole.close(console)  # type: ignore[arg-type]
        stream.close.assert_not_called()
        root.destroy.assert_not_called()

    def test_size_labels_are_stable_for_provenance_table(self) -> None:
        self.assertEqual(app._size_text(0), "0 B")
        self.assertEqual(app._size_text(1024), "1.0 KB")
        self.assertEqual(app._size_text(1024 * 1024), "1.0 MB")

    def test_readiness_blocks_missing_inputs_and_duplicate_launches(self) -> None:
        readiness = mock.Mock()
        button = mock.Mock()
        console = SimpleNamespace(
            selected_workflow_id="online",
            artifacts={
                "cable_model": SimpleNamespace(exists=True, label="Nominal cable model"),
                "sac_policy": SimpleNamespace(exists=False, label="Selected SAC policy"),
            },
            running={},
            readiness_var=readiness,
            launch_button=button,
        )
        app.ResearchConsole._update_readiness(console)  # type: ignore[arg-type]
        readiness.set.assert_called_with(
            "Available; launches in an isolated Python process"
        )
        self.assertEqual(button.configure.call_args.kwargs["state"], app.tk.NORMAL)

        readiness.reset_mock()
        button.reset_mock()
        process = mock.Mock(pid=4321)
        process.poll.return_value = None
        console.artifacts["sac_policy"].exists = True
        console.running["online"] = SimpleNamespace(process=process)
        app.ResearchConsole._update_readiness(console)  # type: ignore[arg-type]
        readiness.set.assert_called_with("Running (PID 4321)")
        self.assertEqual(button.configure.call_args.kwargs["state"], app.tk.DISABLED)

        readiness.reset_mock()
        button.reset_mock()
        console.running = {"policy": SimpleNamespace(process=process)}
        app.ResearchConsole._update_readiness(console)  # type: ignore[arg-type]
        self.assertIn("one heavy workflow at a time", readiness.set.call_args.args[0])
        self.assertEqual(button.configure.call_args.kwargs["state"], app.tk.DISABLED)

    def test_online_mpc_readiness_does_not_require_a_sac_policy(self) -> None:
        readiness = mock.Mock()
        button = mock.Mock()
        console = SimpleNamespace(
            selected_workflow_id="online",
            artifacts={
                "cable_model": SimpleNamespace(
                    exists=True,
                    label="Nominal cable model",
                    status_label="AVAILABLE",
                ),
                "sac_policy": SimpleNamespace(
                    exists=True,
                    label="Selected SAC policy",
                    compatibility="mismatch",
                ),
            },
            running={},
            readiness_var=readiness,
            launch_button=button,
        )
        app.ResearchConsole._update_readiness(console)  # type: ignore[arg-type]
        readiness.set.assert_called_with(
            "Available; launches in an isolated Python process"
        )
        self.assertEqual(button.configure.call_args.kwargs["state"], app.tk.NORMAL)
        self.assertEqual(button.configure.call_args.kwargs["text"], "Launch selected tool")

    def test_launch_selected_passes_command_and_artifact_provenance_to_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = catalog.LaunchSpec(
                workflow_id="online",
                command=(str(Path(sys.executable).resolve()), str(root / "tool.py")),
                cwd=root,
            )
            status = catalog.ArtifactStatus(
                artifact_id="cable_model",
                label="Nominal cable model",
                path=root / "model.json",
                exists=True,
                kind="file",
                item_count=1,
                size_bytes=12,
                modified_text="now",
                sha256="b" * 64,
                schema="model_v1",
                compatibility=None,
            )
            launched = SimpleNamespace(
                process=SimpleNamespace(pid=8675),
                log_path=root / "online.log",
            )
            console = SimpleNamespace(
                selected_workflow_id="online",
                running={},
                artifacts={"cable_model": status},
                last_log_path=None,
                open_log_button=mock.Mock(),
                activity_var=mock.Mock(),
                status_var=mock.Mock(),
                _update_readiness=mock.Mock(),
                root=mock.Mock(),
            )
            with (
                mock.patch.object(app, "build_launch_spec", return_value=spec),
                mock.patch.object(app, "LOG_DIRECTORY", root / "logs"),
                mock.patch.object(
                    app,
                    "launch_workflow_process",
                    return_value=launched,
                ) as launch,
            ):
                app.ResearchConsole.launch_selected(console)  # type: ignore[arg-type]

            _, kwargs = launch.call_args
            manifest = kwargs["manifest"]
            self.assertEqual(manifest["workflow_id"], "online")
            self.assertEqual(manifest["command"], list(spec.command))
            self.assertEqual(manifest["cwd"], str(root))
            artifact = manifest["artifacts_at_launch"]["cable_model"]
            self.assertEqual(artifact["sha256"], "b" * 64)
            self.assertEqual(artifact["schema"], "model_v1")
            self.assertEqual(artifact["status"], "AVAILABLE")
            self.assertIsNone(artifact["compatibility"])
            self.assertIs(console.running["online"], launched)
            self.assertEqual(console.last_log_path, launched.log_path)

    def test_launch_selected_refuses_any_second_heavy_workflow(self) -> None:
        process = mock.Mock(pid=101)
        process.poll.return_value = None
        console = SimpleNamespace(
            selected_workflow_id="online",
            running={"policy": SimpleNamespace(process=process)},
            root=mock.Mock(),
        )
        with (
            mock.patch.object(app.messagebox, "showinfo") as info,
            mock.patch.object(app, "launch_workflow_process") as launch,
        ):
            app.ResearchConsole.launch_selected(console)  # type: ignore[arg-type]
        info.assert_called_once()
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
