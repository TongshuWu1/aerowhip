"""Focused desktop UI for recording and fitting one cable model."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from .config import (
    DEFAULT_CONFIG_PATH,
    load_settings,
    save_cable_settings,
)
from .controller import ControllerEvent, OfflineDderController, SessionPaths
from .planar_data import summarize_planar_observation
from .optimize import MODEL_SCHEMA

RECORDING_INSTRUCTION = (
    "Keep the camera fixed and level and the cable at roughly one distance. Move the "
    "endpoints through slow shapes and faster changes, then pause while it settles."
)


class OfflineDderGui:
    POLL_MS = 100

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.settings = load_settings()
        self.controller = OfflineDderController(self.settings)
        self.camera_manual_selected = False
        self.root.title("Cable Twin - Offline 2D Fitting")
        self.root.geometry("1200x820")
        self.root.minsize(1000, 660)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_style()
        self._build()
        self._load_latest_fit()
        self.refresh()
        self.root.after(0, self.start_preview)
        self.root.after(self.POLL_MS, self.poll)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        background, panel, text = "#0d151f", "#142231", "#e7edf3"
        self.root.configure(background=background)
        style.configure(".", background=background, foreground=text, font=("Segoe UI", 10))
        style.configure("TFrame", background=background)
        style.configure("Card.TFrame", background=panel)
        style.configure("TLabel", background=background, foreground=text)
        style.configure("Card.TLabel", background=panel, foreground=text)
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 22), background=background)
        style.configure("Status.TLabel", font=("Segoe UI Semibold", 12), background=panel)
        style.configure("Good.TLabel", font=("Segoe UI Semibold", 11), background=panel, foreground="#68d391")
        style.configure("Alert.TLabel", font=("Segoe UI Semibold", 11), background=panel, foreground="#f6ad55")
        style.configure("TButton", padding=(13, 8))
        style.configure("Primary.TButton", padding=(16, 9), font=("Segoe UI Semibold", 10))
        style.configure("Treeview", rowheight=29, background="#101c28", fieldbackground="#101c28", foreground=text)
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 10))

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(outer, text="Offline 2D Cable Fitting", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            outer,
            text="RGB centerline + known cable length -> metric 2D trajectory -> fit EI and Cb",
        ).pack(anchor=tk.W, pady=(2, 14))

        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X)
        self.record_button = ttk.Button(
            controls,
            text="1  Record motion",
            style="Primary.TButton",
            command=self.toggle_recording,
        )
        self.record_button.pack(side=tk.LEFT)
        self.review_button = ttk.Button(
            controls,
            text="2  View 2D selected",
            command=self.review_selected,
        )
        self.review_button.pack(side=tk.LEFT, padx=(10, 0))
        self.fit_button = ttk.Button(
            controls,
            text="3  Fit EI + Cb",
            style="Primary.TButton",
            command=self.fit_selected,
        )
        self.fit_button.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(controls, text="Refresh", command=self.refresh).pack(side=tk.RIGHT)
        ttk.Button(controls, text="Delete", command=self.delete_selected).pack(side=tk.RIGHT, padx=(0, 8))

        ttk.Label(outer, text=RECORDING_INSTRUCTION).pack(anchor=tk.W, pady=(8, 0))

        status = ttk.Frame(outer, style="Card.TFrame", padding=14)
        status.pack(fill=tk.X, pady=(14, 12))
        self.status_title = ttk.Label(status, text="Starting camera", style="Status.TLabel")
        self.status_title.pack(anchor=tk.W)
        self.status_detail = ttk.Label(
            status,
            text="Opening the raw ZED recorder and display-only RGB viewport.",
            style="Card.TLabel",
        )
        self.status_detail.pack(anchor=tk.W, pady=(3, 0))

        middle = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        middle.pack(fill=tk.BOTH, expand=True)
        recordings_card = ttk.Frame(middle, style="Card.TFrame", padding=12)
        settings_card = ttk.Frame(middle, style="Card.TFrame", padding=14)
        middle.add(recordings_card, weight=3)
        middle.add(settings_card, weight=2)

        ttk.Label(recordings_card, text="2D fitting recordings", style="Status.TLabel").pack(anchor=tk.W, pady=(0, 8))
        table_frame = ttk.Frame(recordings_card, style="Card.TFrame")
        table_frame.pack(fill=tk.BOTH, expand=True)
        self.recordings = ttk.Treeview(
            table_frame,
            columns=("duration", "evidence", "state"),
            selectmode="extended",
        )
        self.recordings.heading("#0", text="Recording")
        self.recordings.heading("duration", text="Duration")
        self.recordings.heading("evidence", text="Complete 2D")
        self.recordings.heading("state", text="Status")
        self.recordings.column("#0", width=315, anchor=tk.W)
        self.recordings.column("duration", width=85, anchor=tk.CENTER)
        self.recordings.column("evidence", width=115, anchor=tk.CENTER)
        self.recordings.column("state", width=95, anchor=tk.CENTER)
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.recordings.yview)
        self.recordings.configure(yscrollcommand=scrollbar.set)
        self.recordings.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(settings_card, text="Cable specimen", style="Status.TLabel").grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        self.cable_var = tk.StringVar(value="1")
        self.length_var = tk.StringVar(value=f"{self.settings.cable.length_m:g}")
        self.mass_var = tk.StringVar(value=f"{self.settings.cable.mass_kg:g}")
        self.diameter_var = tk.StringVar(value=f"{self.settings.cable.diameter_m:g}")
        self.iterations_var = tk.StringVar(
            value=str(self.settings.optimization.optimizer_iterations)
        )
        self._entry_row(settings_card, 1, "Cable identity", self.cable_var, "1 or 2")
        self._entry_row(settings_card, 2, "Active length", self.length_var, "m")
        self._entry_row(settings_card, 3, "Mass", self.mass_var, "kg")
        self._entry_row(settings_card, 4, "Diameter", self.diameter_var, "m")
        self._entry_row(
            settings_card, 5, "Optimizer iterations", self.iterations_var, ""
        )
        ttk.Button(settings_card, text="Save cable values", command=self.save_cable).grid(row=6, column=0, columnspan=3, sticky=tk.EW, pady=(12, 18))

        ttk.Separator(settings_card).grid(row=7, column=0, columnspan=3, sticky=tk.EW, pady=(0, 14))
        ttk.Label(settings_card, text="Camera", style="Status.TLabel").grid(row=8, column=0, columnspan=3, sticky=tk.W)
        self.exposure_var = tk.StringVar(value="30")
        self.gain_var = tk.StringVar(value="1")
        self.camera_mode_var = tk.StringVar(value="Opening camera")
        self._entry_row(settings_card, 9, "Exposure", self.exposure_var, "0-100")
        self._entry_row(settings_card, 10, "Gain", self.gain_var, "0-100")
        camera_buttons = ttk.Frame(settings_card, style="Card.TFrame")
        camera_buttons.grid(row=11, column=0, columnspan=3, sticky=tk.EW, pady=(8, 4))
        self.camera_apply_button = ttk.Button(
            camera_buttons,
            text="Apply manual",
            command=self.apply_manual_camera,
            state=tk.DISABLED,
        )
        self.camera_apply_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.camera_auto_button = ttk.Button(
            camera_buttons,
            text="Auto",
            command=self.apply_automatic_camera,
            state=tk.DISABLED,
        )
        self.camera_auto_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
        self._value_row(settings_card, 12, "Active mode", self.camera_mode_var)

        ttk.Separator(settings_card).grid(row=13, column=0, columnspan=3, sticky=tk.EW, pady=(12, 14))
        ttk.Label(settings_card, text="Image-plane assumption", style="Status.TLabel").grid(row=14, column=0, columnspan=3, sticky=tk.W)
        self.metric_scale_var = tk.StringVar(value="From known cable length")
        self._value_row(settings_card, 15, "Metric scale", self.metric_scale_var)

        ttk.Separator(settings_card).grid(row=16, column=0, columnspan=3, sticky=tk.EW, pady=(12, 14))
        ttk.Label(settings_card, text="2D fit result", style="Status.TLabel").grid(row=17, column=0, columnspan=3, sticky=tk.W)
        self.model_status_var = tk.StringVar(value="NOT FITTED")
        self.model_status_label = ttk.Label(
            settings_card,
            textvariable=self.model_status_var,
            style="Alert.TLabel",
        )
        self.model_status_label.grid(row=18, column=0, columnspan=3, sticky=tk.W, pady=(8, 4))
        self.ei_var = tk.StringVar(value="-")
        self.damping_var = tk.StringVar(value="-")
        self.reprojection_var = tk.StringVar(value="-")
        self.rollout_var = tk.StringVar(value="-")
        self._value_row(settings_card, 19, "Bending stiffness EI", self.ei_var)
        self._value_row(settings_card, 20, "Bending damping Cb", self.damping_var)
        self._value_row(settings_card, 21, "RGB measurement scale", self.reprojection_var)
        self._value_row(settings_card, 22, "Rod rollout residual", self.rollout_var)
        settings_card.columnconfigure(1, weight=1)

        log_card = ttk.Frame(outer, style="Card.TFrame", padding=(12, 8))
        log_card.pack(fill=tk.X, pady=(12, 0))
        self.log = tk.Text(
            log_card,
            height=4,
            bg="#101c28",
            fg="#c9d7e3",
            insertbackground="#c9d7e3",
            relief=tk.FLAT,
            font=("Cascadia Mono", 9),
            state=tk.DISABLED,
        )
        self.log.pack(fill=tk.X)

    @staticmethod
    def _entry_row(parent, row, label, variable, unit) -> None:
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky=tk.W, pady=4)
        ttk.Entry(parent, textvariable=variable, width=14).grid(row=row, column=1, sticky=tk.EW, padx=(12, 8), pady=4)
        ttk.Label(parent, text=unit, style="Card.TLabel").grid(row=row, column=2, sticky=tk.W)

    @staticmethod
    def _value_row(parent, row, label, variable) -> None:
        ttk.Label(parent, text=label, style="Card.TLabel").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Label(parent, textvariable=variable, style="Card.TLabel").grid(row=row, column=1, columnspan=2, sticky=tk.E, pady=5)

    def set_status(self, title: str, detail: str) -> None:
        self.status_title.configure(text=title)
        self.status_detail.configure(text=detail)

    def append_log(self, message: str) -> None:
        if not message:
            return
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message.rstrip() + "\n")
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 160:
            self.log.delete("1.0", f"{lines - 120}.0")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _read_values(self) -> tuple[float, float, float, int, int]:
        length, mass, diameter = map(float, (self.length_var.get(), self.mass_var.get(), self.diameter_var.get()))
        identity = int(self.cable_var.get())
        iterations = int(self.iterations_var.get())
        if any(not math.isfinite(value) or value <= 0.0 for value in (length, mass, diameter)):
            raise ValueError("Cable length, mass, and diameter must be positive.")
        if identity not in (1, 2) or iterations < 3:
            raise ValueError(
                "Cable identity must be 1 or 2 and optimizer iterations must be at least 3."
            )
        return length, mass, diameter, identity, iterations

    def save_cable(self) -> bool:
        try:
            length, mass, diameter, _identity, _iterations = self._read_values()
            save_cable_settings(DEFAULT_CONFIG_PATH, length_m=length, mass_kg=mass, diameter_m=diameter)
            self.settings = load_settings()
            self.controller.settings = self.settings
            self.refresh()
        except (ValueError, OSError) as error:
            messagebox.showerror("Cable values", str(error), parent=self.root)
            return False
        self.set_status("Cable values saved", "The next fit will use these measured values.")
        return True

    def start_preview(self) -> None:
        if self.controller.active_task is not None:
            return
        try:
            if self.camera_manual_selected:
                exposure, gain = self._read_camera_values()
                self.controller.start_preview(
                    manual_exposure=exposure,
                    manual_gain=gain,
                )
            else:
                self.controller.start_preview()
        except Exception as error:
            self.set_status("Camera unavailable", str(error))
            self.append_log(str(error))

    def _read_camera_values(self) -> tuple[int, int]:
        exposure, gain = int(self.exposure_var.get()), int(self.gain_var.get())
        if not 0 <= exposure <= 100 or not 0 <= gain <= 100:
            raise ValueError("Exposure and gain must be integers between 0 and 100.")
        return exposure, gain

    def apply_manual_camera(self) -> None:
        try:
            exposure, gain = self._read_camera_values()
            self.controller.set_camera_manual(exposure, gain)
        except (TypeError, ValueError, RuntimeError) as error:
            messagebox.showerror("Camera settings", str(error), parent=self.root)
            return
        self.record_button.configure(state=tk.DISABLED)
        self.camera_apply_button.configure(state=tk.DISABLED)
        self.camera_auto_button.configure(state=tk.DISABLED)
        self.set_status("Applying camera settings", "Waiting for verified ZED readback.")

    def apply_automatic_camera(self) -> None:
        try:
            self.controller.set_camera_automatic()
        except RuntimeError as error:
            messagebox.showerror("Camera settings", str(error), parent=self.root)
            return
        self.record_button.configure(state=tk.DISABLED)
        self.camera_apply_button.configure(state=tk.DISABLED)
        self.camera_auto_button.configure(state=tk.DISABLED)
        self.set_status("Applying automatic exposure", "Waiting for verified ZED readback.")

    def toggle_recording(self) -> None:
        if self.controller.is_recording:
            self.controller.request_stop_recording()
            self.record_button.configure(state=tk.DISABLED, text="Saving...")
            return
        try:
            _length, _mass, _diameter, identity, _iterations = self._read_values()
            self.controller.start_recording(identity)
        except Exception as error:
            messagebox.showerror("Recording", str(error), parent=self.root)
            return
        self.record_button.configure(state=tk.DISABLED, text="Starting...")
        self.camera_apply_button.configure(state=tk.DISABLED)
        self.camera_auto_button.configure(state=tk.DISABLED)
        self.set_status("Starting recording", RECORDING_INSTRUCTION)

    def selected_sessions(self) -> list[SessionPaths]:
        return [self.controller.recording_artifacts(Path(iid)) for iid in self.recordings.selection()]

    def fit_selected(self) -> None:
        sessions = self.selected_sessions()
        if not sessions:
            messagebox.showinfo("Fit EI + Cb", "Select one or more recordings.", parent=self.root)
            return
        try:
            if not self.save_cable():
                return
            _length, _mass, _diameter, identity, iterations = self._read_values()
            if any(self.controller.cable_identity(item) != identity for item in sessions):
                raise ValueError("Selected recordings do not match the cable identity shown on the right.")
            self.controller.start_fit(sessions, iterations=iterations)
        except Exception as error:
            messagebox.showerror("Fit EI + Cb", str(error), parent=self.root)
            return
        self._set_busy(True)
        self.set_status(
            "Fitting cable model",
            "Extracting planar PIDNet curves, then estimating EI and Cb.",
        )

    def review_selected(self) -> None:
        sessions = self.selected_sessions()
        if len(sessions) != 1:
            messagebox.showinfo(
                "View 2D trajectory",
                "Select exactly one recording.",
                parent=self.root,
            )
            return
        processing = not (
            self.controller.has_current_observations(sessions[0])
            and self.controller.has_current_trajectory(sessions[0])
        )
        try:
            self.controller.start_2d_view(sessions[0].svo)
        except Exception as error:
            messagebox.showerror("View 2D trajectory", str(error), parent=self.root)
            return
        self._set_busy(True)
        if processing:
            self.set_status(
                "Extracting 2D trajectory",
                "PIDNet is extracting the endpoint-to-endpoint route and applying one known-length image scale.",
            )
        else:
            self.set_status(
                "2D trajectory viewer open",
                "The RGB centerline and the unfiltered 24 PIDNet fitting nodes are shown together.",
            )

    def delete_selected(self) -> None:
        sessions = self.selected_sessions()
        if not sessions:
            return
        if not messagebox.askyesno("Delete recordings", f"Permanently delete {len(sessions)} selected recording(s) and derived files?", parent=self.root):
            return
        try:
            deleted = self.controller.delete_recordings([item.svo for item in sessions])
        except Exception as error:
            messagebox.showerror("Delete recordings", str(error), parent=self.root)
            return
        self.refresh()
        self.set_status("Recordings deleted", f"Removed {len(deleted)} files. This cannot be recovered from the app.")

    def _set_busy(self, busy: bool) -> None:
        state = tk.DISABLED if busy else tk.NORMAL
        for button in (
            self.record_button,
            self.fit_button,
            self.review_button,
            self.camera_apply_button,
            self.camera_auto_button,
        ):
            button.configure(state=state)

    def refresh(self, selected: Path | None = None) -> None:
        previous = set(self.recordings.selection())
        self.recordings.delete(*self.recordings.get_children())
        for svo in sorted(self.settings.recording_directory.glob("*.svo2"), key=lambda item: item.stat().st_mtime, reverse=True):
            session = self.controller.recording_artifacts(svo)
            duration, evidence, state = "-", "-", "Raw video"
            if self.controller.has_current_observations(session):
                try:
                    summary = summarize_planar_observation(session.observations)
                    duration = f"{summary.duration_s:.1f} s"
                    evidence = f"{100.0 * summary.complete_fraction:.0f}%"
                    state = "2D ready" if summary.fitted else "2D extracted"
                except ValueError:
                    state = "Process needed"
            iid = str(svo.resolve())
            self.recordings.insert(
                "", tk.END, iid=iid, text=svo.name,
                values=(duration, evidence, state),
            )
        target = str(selected.resolve()) if selected is not None else None
        children = set(self.recordings.get_children())
        if target in children:
            self.recordings.selection_set(target)
        else:
            retained = list(previous & children)
            if retained:
                self.recordings.selection_set(retained)

    def _show_model(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != MODEL_SCHEMA:
            raise ValueError("Saved model schema is incompatible.")
        optimized, fit = payload["optimized"], payload["fit"]
        self.model_status_var.set("FITTED")
        self.model_status_label.configure(style="Good.TLabel")
        self.ei_var.set(f"{float(optimized['bending_stiffness_n_m2']):.6g} N m^2")
        self.damping_var.set(
            f"{float(optimized['bending_damping_n_m2_s']):.4g} N m^2 s"
        )
        self.reprojection_var.set(
            f"{1000.0 * float(fit['observation_curve_residual_m']):.2f} mm"
        )
        self.rollout_var.set(
            f"{1000.0 * float(fit['train_window_curve_residual_m']):.2f} mm"
        )

    def _load_latest_fit(self) -> None:
        candidates = sorted(
            self.settings.model_directory.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            try:
                self._show_model(path)
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                continue
            return

    def handle_event(self, event: ControllerEvent) -> None:
        if event.kind == "log":
            self.append_log(event.text)
        elif event.kind == "controller_error":
            self.append_log(event.text)
            self.set_status("Recorder error", event.text)
        elif event.kind == "preview_ready":
            self.record_button.configure(state=tk.NORMAL, text="1  Record motion")
            self.camera_apply_button.configure(state=tk.NORMAL)
            self.camera_auto_button.configure(state=tk.NORMAL)
            self._show_camera_settings(event)
            self.set_status(
                "Ready to record",
                "Use the RGB view for framing. Keep the cable approximately parallel to the image plane.",
            )
        elif event.kind == "capture_settings":
            self._show_camera_settings(event)
            self.record_button.configure(state=tk.NORMAL)
            self.camera_apply_button.configure(state=tk.NORMAL)
            self.camera_auto_button.configure(state=tk.NORMAL)
            self.set_status(
                "Camera settings applied",
                f"{self.camera_mode_var.get()}; verify cable sharpness in the live RGB view.",
            )
        elif event.kind == "recording_started":
            self.record_button.configure(state=tk.NORMAL, text="Stop recording")
            self.camera_apply_button.configure(state=tk.DISABLED)
            self.camera_auto_button.configure(state=tk.DISABLED)
            self.set_status("Recording", RECORDING_INSTRUCTION)
        elif event.kind == "recording_finished":
            self.record_button.configure(state=tk.NORMAL, text="1  Record motion")
            self.camera_apply_button.configure(state=tk.NORMAL)
            self.camera_auto_button.configure(state=tk.NORMAL)
            if event.return_code == 0 and event.output_path is not None and event.output_path.is_file():
                self.refresh(event.output_path)
                self.set_status(
                    "Recording saved",
                    f"Saved {int(event.frame_count or 0)} lossless ZED frames. Select it to view the 2D trajectory or fit directly.",
                )
            else:
                self.set_status("Recording failed", "The recorder closed before publishing a complete SVO2.")
        elif event.kind == "finished":
            self._set_busy(False)
            if event.task == "fit":
                if event.return_code == 0 and event.output_path is not None and event.output_path.is_file():
                    self._show_model(event.output_path)
                    self.refresh()
                    self.set_status(
                        "Model fitted",
                        "The fitted rod parameters are shown on the right.",
                    )
                else:
                    self._load_latest_fit()
                    self.set_status("Model fit failed", "The optimizer did not produce a valid EI and Cb model.")
                self.start_preview()
            elif event.task == "process_2d":
                if (
                    event.return_code == 0
                    and event.output_path is not None
                    and event.output_path.is_file()
                ):
                    self.refresh()
                    self.set_status(
                        "2D trajectory ready",
                        "The unfiltered image-plane PIDNet nodes were saved and can now be fitted.",
                    )
                else:
                    self.set_status(
                        "2D processing failed",
                        "Read the log below; no partial 2D trajectory was kept.",
                    )
                self.start_preview()
            elif event.task == "view_2d":
                self.set_status("2D viewer closed", "The recorder is ready again.")
                self.start_preview()
            elif event.task == "preview" and event.return_code != 0:
                self.set_status("Camera closed", "Restart the application after the camera becomes available.")

    def _show_camera_settings(self, event: ControllerEvent) -> None:
        if event.camera_exposure is None or event.camera_gain is None:
            raise ValueError("Camera readback is incomplete.")
        self.exposure_var.set(str(event.camera_exposure))
        self.gain_var.set(str(event.camera_gain))
        self.camera_manual_selected = not bool(event.camera_automatic)
        self.camera_mode_var.set(
            "Automatic" if event.camera_automatic else "Manual"
        )

    def poll(self) -> None:
        for event in self.controller.poll():
            try:
                self.handle_event(event)
            except Exception as error:
                self.append_log(str(error))
                self.set_status("Application error", str(error))
        self.root.after(self.POLL_MS, self.poll)

    def close(self) -> None:
        if self.controller.is_recording:
            messagebox.showinfo("Recording active", "Stop the recording and wait for it to save before closing.", parent=self.root)
            return
        self.controller.shutdown()
        self.root.destroy()


def run_gui() -> None:
    root = tk.Tk()
    OfflineDderGui(root)
    root.mainloop()
