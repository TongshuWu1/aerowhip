"""Research-oriented launcher for the cable-whip workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import tkinter as tk
from tkinter import messagebox, ttk
from typing import IO

from .catalog import (
    ArtifactStatus,
    LaunchSpec,
    REPOSITORY_DIRECTORY,
    WORKFLOWS,
    build_launch_spec,
    inspect_artifacts,
    workflow_by_id,
)


LOG_DIRECTORY = REPOSITORY_DIRECTORY / "data" / "research_console" / "logs"


@dataclass(slots=True)
class RunningWorkflow:
    process: subprocess.Popen[bytes]
    log_stream: IO[bytes]
    log_path: Path


def launch_workflow_process(
    spec: LaunchSpec,
    log_path: str | Path,
    *,
    manifest: dict[str, object] | None = None,
) -> RunningWorkflow:
    """Start one isolated GUI process without a shell or captured pipe."""

    destination = Path(log_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = destination.open("ab", buffering=0)
    if manifest is not None:
        header = (
            "# CABLE WHIP RESEARCH SESSION\n"
            + json.dumps(manifest, indent=2, sort_keys=True)
            + "\n# CHILD PROCESS OUTPUT\n"
        )
        stream.write(header.encode("utf-8"))
    creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0
    try:
        process = subprocess.Popen(
            list(spec.command),
            cwd=str(spec.cwd),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            shell=False,
            creationflags=creation_flags,
        )
    except BaseException:
        stream.close()
        raise
    return RunningWorkflow(process=process, log_stream=stream, log_path=destination)


def _size_text(size_bytes: int) -> str:
    value = float(max(0, size_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return "--"


def _open_directory(path: Path, *, directory_hint: bool = False) -> None:
    destination = path if path.is_dir() or directory_hint else path.parent
    destination.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(destination))  # type: ignore[attr-defined]
        return
    subprocess.Popen(("xdg-open", str(destination)), start_new_session=True)


class ResearchConsole:
    POLL_MS = 500

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Cable Whip Research Console")
        self.root.geometry("1360x840")
        self.root.minsize(1100, 700)
        self.root.configure(background="#ffffff")
        self._configure_style()

        self.artifacts: dict[str, ArtifactStatus] = {}
        self.running: dict[str, RunningWorkflow] = {}
        self.workflow_buttons: dict[str, ttk.Button] = {}
        self.selected_workflow_id = "online"

        self.stage_var = tk.StringVar()
        self.title_var = tk.StringVar()
        self.objective_var = tk.StringVar()
        self.method_var = tk.StringVar()
        self.inputs_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.readiness_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.activity_var = tk.StringVar(value="No tools launched in this session")
        self.artifact_path_var = tk.StringVar(value="Select an artifact to inspect its path")
        self.last_log_path: Path | None = None

        self._build()
        self.refresh_artifacts()
        self.select_workflow(self.selected_workflow_id)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(self.POLL_MS, self._poll_processes)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        white = "#ffffff"
        text = "#171717"
        muted = "#5f6368"
        border = "#d5d9dd"
        surface = "#f5f6f7"
        accent = "#235f91"
        accent_hover = "#194b73"

        style.configure(".", background=white, foreground=text, font=("Segoe UI", 10))
        style.configure("TFrame", background=white)
        style.configure(
            "Panel.TFrame",
            background=white,
            borderwidth=1,
            relief=tk.SOLID,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.configure("TLabel", background=white, foreground=text)
        style.configure("Muted.TLabel", foreground=muted)
        style.configure("Status.TLabel", foreground=accent, font=("Segoe UI Semibold", 9))
        style.configure(
            "Title.TLabel",
            font=("Segoe UI Semibold", 21),
            foreground="#101010",
        )
        style.configure(
            "WorkflowTitle.TLabel",
            font=("Segoe UI Semibold", 17),
            foreground="#101010",
        )
        style.configure(
            "Section.TLabel",
            font=("Segoe UI Semibold", 9),
            foreground="#34373a",
        )
        style.configure(
            "TButton",
            background=surface,
            foreground=text,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            focuscolor=border,
            padding=(10, 7),
        )
        style.map(
            "TButton",
            background=[("active", "#eceeef"), ("disabled", "#fafafa")],
            foreground=[("disabled", "#999999")],
        )
        style.configure(
            "Primary.TButton",
            background=accent,
            foreground=white,
            bordercolor=accent,
            lightcolor=accent,
            darkcolor=accent,
            focuscolor=accent,
            padding=(14, 8),
        )
        style.map(
            "Primary.TButton",
            background=[("active", accent_hover), ("disabled", "#aebdca")],
            foreground=[("disabled", white)],
        )
        style.configure(
            "Step.TButton",
            background=white,
            foreground=text,
            anchor=tk.CENTER,
            padding=(14, 10),
        )
        style.configure(
            "Selected.Step.TButton",
            background="#eaf2f8",
            foreground="#174d75",
            bordercolor=accent,
            lightcolor=accent,
            darkcolor=accent,
            anchor=tk.CENTER,
            padding=(14, 10),
        )
        style.map(
            "Selected.Step.TButton",
            background=[("active", "#dceaf4")],
        )
        style.configure("TSeparator", background=border)
        style.configure(
            "Treeview",
            background=white,
            fieldbackground=white,
            foreground=text,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            rowheight=27,
        )
        style.configure(
            "Treeview.Heading",
            background="#f1f2f3",
            foreground="#252525",
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            font=("Segoe UI Semibold", 9),
            padding=(7, 6),
        )
        style.map(
            "Treeview",
            background=[("selected", "#dceaf4")],
            foreground=[("selected", text)],
        )

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=(22, 18))
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X, pady=(0, 10))
        text = ttk.Frame(header)
        text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(text, text="Cable Whip Research Console", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            text,
            text="Experimental workflow, computational tools, and artifact provenance",
            style="Muted.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))
        actions = ttk.Frame(header)
        actions.pack(side=tk.RIGHT)
        ttk.Button(actions, text="Refresh provenance", command=self.refresh_artifacts).pack(side=tk.LEFT)
        ttk.Button(
            actions,
            text="Open project data",
            command=lambda: _open_directory(REPOSITORY_DIRECTORY / "data"),
        ).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Separator(outer).pack(fill=tk.X, pady=(0, 12))

        limitation = tk.Frame(
            outer,
            background="#f5f6f7",
            highlightbackground="#d5d9dd",
            highlightthickness=1,
        )
        limitation.pack(fill=tk.X, pady=(0, 15))
        tk.Frame(limitation, width=4, background="#235f91").pack(side=tk.LEFT, fill=tk.Y)
        boundary_text = tk.Frame(limitation, background="#f5f6f7", padx=11, pady=8)
        boundary_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            boundary_text,
            text="CURRENT RESEARCH SCOPE",
            background="#f5f6f7",
            foreground="#235f91",
            font=("Segoe UI Semibold", 9),
        ).pack(anchor=tk.W)
        tk.Label(
            boundary_text,
            text=(
                "Full ordered marker-position feedback with perfect association; "
                "online adaptation is not enabled; the plant is acceleration-tracked and not force-coupled."
            ),
            background="#f5f6f7",
            foreground="#242424",
            font=("Segoe UI", 9),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(1, 0))

        workflow_heading = ttk.Frame(outer)
        workflow_heading.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(
            workflow_heading,
            text="RESEARCH WORKFLOW",
            style="Section.TLabel",
        ).pack(side=tk.LEFT)
        ttk.Label(
            workflow_heading,
            text="Select a stage to review its method and launch its dedicated tool.",
            style="Muted.TLabel",
        ).pack(side=tk.LEFT, padx=(12, 0))

        workflow_bar = ttk.Frame(outer)
        workflow_bar.pack(fill=tk.X, pady=(0, 15))
        primary_workflows = tuple(item for item in WORKFLOWS if item.primary)
        for index, workflow in enumerate(primary_workflows):
            button = ttk.Button(
                workflow_bar,
                text=f"{workflow.stage:02d}  {workflow.short_title}",
                style="Step.TButton",
                command=lambda key=workflow.workflow_id: self.select_workflow(key),
            )
            button.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.workflow_buttons[workflow.workflow_id] = button
            if index < len(primary_workflows) - 1:
                ttk.Label(workflow_bar, text="→", style="Muted.TLabel").pack(
                    side=tk.LEFT,
                    padx=7,
                )
        ttk.Separator(workflow_bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT,
            fill=tk.Y,
            padx=12,
        )
        for workflow in (item for item in WORKFLOWS if not item.primary):
            button = ttk.Button(
                workflow_bar,
                text=f"Utility  ·  {workflow.short_title}",
                style="Step.TButton",
                command=lambda key=workflow.workflow_id: self.select_workflow(key),
            )
            button.pack(side=tk.LEFT)
            self.workflow_buttons[workflow.workflow_id] = button

        body = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        definition_panel = ttk.Frame(body, style="Panel.TFrame", padding=17)
        provenance_panel = ttk.Frame(body, style="Panel.TFrame", padding=17)
        body.add(definition_panel, weight=2)
        body.add(provenance_panel, weight=3)

        ttk.Label(
            definition_panel,
            textvariable=self.stage_var,
            style="Status.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            definition_panel,
            textvariable=self.title_var,
            style="WorkflowTitle.TLabel",
        ).pack(anchor=tk.W, pady=(3, 7))
        ttk.Label(
            definition_panel,
            textvariable=self.objective_var,
            wraplength=470,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X)

        ttk.Separator(definition_panel).pack(fill=tk.X, pady=(14, 8))
        self._definition_row(definition_panel, "METHOD", self.method_var)
        self._definition_row(definition_panel, "INPUTS", self.inputs_var)
        self._definition_row(definition_panel, "OUTPUT", self.output_var)

        ttk.Separator(definition_panel).pack(fill=tk.X, pady=(9, 12))
        launch_row = ttk.Frame(definition_panel)
        launch_row.pack(fill=tk.X)
        self.launch_button = ttk.Button(
            launch_row,
            text="Launch selected tool",
            style="Primary.TButton",
            command=self.launch_selected,
        )
        self.launch_button.pack(side=tk.LEFT)
        ttk.Label(
            definition_panel,
            textvariable=self.readiness_var,
            style="Status.TLabel",
            wraplength=470,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X, pady=(9, 0))

        session = ttk.Frame(definition_panel)
        session.pack(fill=tk.X, side=tk.BOTTOM, pady=(14, 0))
        ttk.Separator(session).pack(fill=tk.X, pady=(0, 10))
        ttk.Label(session, text="SESSION ACTIVITY", style="Section.TLabel").pack(anchor=tk.W)
        ttk.Label(
            session,
            textvariable=self.activity_var,
            style="Muted.TLabel",
            wraplength=470,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X, pady=(4, 7))
        self.open_log_button = ttk.Button(
            session,
            text="Open session log",
            command=self.open_session_log,
            state=tk.DISABLED,
        )
        self.open_log_button.pack(anchor=tk.W)

        provenance_header = ttk.Frame(provenance_panel)
        provenance_header.pack(fill=tk.X, pady=(0, 9))
        provenance_text = ttk.Frame(provenance_header)
        provenance_text.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(
            provenance_text,
            text="ARTIFACT PROVENANCE",
            style="Section.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            provenance_text,
            text="Availability, compatibility, schema, and content hash at the canonical project paths.",
            style="Muted.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        table_frame = ttk.Frame(provenance_panel)
        table_frame.pack(fill=tk.BOTH, expand=True)
        self.artifact_table = ttk.Treeview(
            table_frame,
            columns=("status", "artifact", "items", "modified", "hash", "schema"),
            show="headings",
            selectmode="browse",
        )
        headings = (
            ("status", "Status", 105, tk.CENTER),
            ("artifact", "Artifact", 210, tk.W),
            ("items", "Items / size", 110, tk.E),
            ("modified", "Modified", 135, tk.W),
            ("hash", "SHA-256", 95, tk.W),
            ("schema", "Schema / contract", 270, tk.W),
        )
        for column, label, width, anchor in headings:
            self.artifact_table.heading(column, text=label)
            self.artifact_table.column(column, width=width, anchor=anchor, stretch=column in ("artifact", "schema"))
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.artifact_table.yview)
        self.artifact_table.configure(yscrollcommand=scrollbar.set)
        self.artifact_table.tag_configure("missing", foreground="#8f2f2f")
        self.artifact_table.tag_configure("provisional", foreground="#7a5700")
        self.artifact_table.tag_configure("compatible", foreground="#235f91")
        self.artifact_table.tag_configure("mismatch", foreground="#9b2929")
        self.artifact_table.tag_configure("available", foreground="#252525")
        self.artifact_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.artifact_table.bind("<<TreeviewSelect>>", self._artifact_selected)

        path_row = ttk.Frame(provenance_panel)
        path_row.pack(fill=tk.X, pady=(9, 0))
        ttk.Label(
            path_row,
            textvariable=self.artifact_path_var,
            style="Muted.TLabel",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_row, text="Open location", command=self.open_selected_artifact).pack(side=tk.RIGHT)

        ttk.Separator(outer).pack(fill=tk.X, pady=(13, 7))
        footer = ttk.Frame(outer)
        footer.pack(fill=tk.X)
        ttk.Label(footer, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            footer,
            text=str(REPOSITORY_DIRECTORY),
            style="Muted.TLabel",
        ).pack(side=tk.RIGHT)

    @staticmethod
    def _definition_row(parent: ttk.Frame, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=(4, 6))
        ttk.Label(row, text=label, style="Section.TLabel").pack(anchor=tk.W)
        ttk.Label(
            row,
            textvariable=variable,
            wraplength=470,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, fill=tk.X, pady=(2, 0))

    def select_workflow(self, workflow_id: str) -> None:
        workflow = workflow_by_id(workflow_id)
        self.selected_workflow_id = workflow_id
        for key, button in self.workflow_buttons.items():
            button.configure(
                style="Selected.Step.TButton" if key == workflow_id else "Step.TButton"
            )
        self.stage_var.set(
            f"STAGE {workflow.stage:02d}" if workflow.primary else "METHOD UTILITY"
        )
        self.title_var.set(workflow.title)
        self.objective_var.set(workflow.objective)
        self.method_var.set(workflow.method)
        self.inputs_var.set("  •  ".join(workflow.inputs))
        self.output_var.set(workflow.output)
        self._update_readiness()

    def _update_readiness(self) -> None:
        workflow = workflow_by_id(self.selected_workflow_id)
        missing = [
            key for key in workflow.required_artifacts
            if key not in self.artifacts or not self.artifacts[key].exists
        ]
        active = self.running.get(workflow.workflow_id)
        any_active = next(
            (
                (key, item)
                for key, item in self.running.items()
                if item.process.poll() is None
            ),
            None,
        )
        if active is not None and active.process.poll() is None:
            self.readiness_var.set(f"Running (PID {active.process.pid})")
            self.launch_button.configure(state=tk.DISABLED, text="Tool is running")
        elif any_active is not None:
            running_workflow = workflow_by_id(any_active[0])
            self.readiness_var.set(
                f"Busy: {running_workflow.short_title} is running; one heavy workflow at a time"
            )
            self.launch_button.configure(state=tk.DISABLED, text="Another tool is running")
        elif missing:
            labels = [self.artifacts[key].label if key in self.artifacts else key for key in missing]
            self.readiness_var.set("Blocked: missing " + ", ".join(labels))
            self.launch_button.configure(state=tk.DISABLED, text="Requirements missing")
        else:
            model = self.artifacts.get("cable_model")
            qualifier = "Available"
            if workflow.workflow_id == "identify":
                csv_status = self.artifacts.get("optitrack_csv")
                if csv_status is None or not csv_status.exists:
                    qualifier = "Available — no OptiTrack CSVs are currently indexed"
            elif model is not None and getattr(model, "status_label", "") == "PROVISIONAL":
                qualifier = "Available with provisional transferred model"
            self.readiness_var.set(qualifier + "; launches in an isolated Python process")
            self.launch_button.configure(state=tk.NORMAL, text="Launch selected tool")

    def refresh_artifacts(self) -> None:
        self.status_var.set("Inspecting canonical artifacts…")
        self.root.update_idletasks()
        try:
            statuses = inspect_artifacts()
        except Exception as error:
            self.status_var.set(f"Artifact inspection failed: {error}")
            return
        self.artifacts = {status.artifact_id: status for status in statuses}
        children = self.artifact_table.get_children() if hasattr(self, "artifact_table") else ()
        for item in children:
            self.artifact_table.delete(item)
        if hasattr(self, "artifact_table"):
            for status in statuses:
                items = (
                    _size_text(status.size_bytes)
                    if status.kind == "file"
                    else f"{status.item_count}  /  {_size_text(status.size_bytes)}"
                )
                self.artifact_table.insert(
                    "",
                    tk.END,
                    iid=status.artifact_id,
                    tags=(status.status_label.lower(),),
                    values=(
                        status.status_label,
                        status.label,
                        items,
                        status.modified_text,
                        status.short_hash,
                        status.schema or "--",
                    ),
                )
        self.status_var.set(f"Provenance refreshed at {datetime.now().strftime('%H:%M:%S')}")
        if hasattr(self, "launch_button"):
            self._update_readiness()

    def launch_selected(self) -> None:
        workflow = workflow_by_id(self.selected_workflow_id)
        other_active = next(
            (
                (key, item)
                for key, item in self.running.items()
                if item.process.poll() is None
            ),
            None,
        )
        if other_active is not None:
            running_workflow = workflow_by_id(other_active[0])
            messagebox.showinfo(
                "Research tool already running",
                f"{running_workflow.title} is still running. Close it before launching another heavy workflow.",
                parent=self.root,
            )
            return
        active = self.running.get(workflow.workflow_id)
        if active is not None and active.process.poll() is None:
            messagebox.showinfo("Already running", f"{workflow.title} is already open.", parent=self.root)
            return
        try:
            spec = build_launch_spec(workflow.workflow_id)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            artifact_manifest = {
                artifact_id: {
                    "path": str(status.path),
                    "status": status.status_label,
                    "sha256": status.sha256,
                    "schema": status.schema,
                    "compatibility": status.compatibility,
                }
                for artifact_id, status in sorted(self.artifacts.items())
            }
            launched = launch_workflow_process(
                spec,
                LOG_DIRECTORY / f"{workflow.workflow_id}_{timestamp}.log",
                manifest={
                    "created_local": datetime.now().astimezone().isoformat(),
                    "workflow_id": workflow.workflow_id,
                    "workflow_title": workflow.title,
                    "command": list(spec.command),
                    "cwd": str(spec.cwd),
                    "artifacts_at_launch": artifact_manifest,
                },
            )
        except Exception as error:
            messagebox.showerror("Could not launch tool", str(error), parent=self.root)
            self.status_var.set(f"Launch failed: {error}")
            return
        self.running[workflow.workflow_id] = launched
        self.last_log_path = launched.log_path
        self.open_log_button.configure(state=tk.NORMAL)
        self.activity_var.set(
            f"{workflow.short_title} running\nPID {launched.process.pid}\nLog: {launched.log_path.name}"
        )
        self.status_var.set(f"Launched {workflow.title}")
        self._update_readiness()

    def _poll_processes(self) -> None:
        completed: list[tuple[str, int]] = []
        for workflow_id, running in tuple(self.running.items()):
            return_code = running.process.poll()
            if return_code is not None:
                running.log_stream.close()
                completed.append((workflow_id, int(return_code)))
                del self.running[workflow_id]
        if completed:
            workflow_id, return_code = completed[-1]
            workflow = workflow_by_id(workflow_id)
            self.activity_var.set(
                f"{workflow.short_title} closed with exit code {return_code}."
            )
            self.status_var.set(
                f"{workflow.title} closed normally"
                if return_code == 0
                else f"{workflow.title} failed; inspect its session log"
            )
            self._update_readiness()
            self.refresh_artifacts()
        self.root.after(self.POLL_MS, self._poll_processes)

    def _artifact_selected(self, _event: object | None = None) -> None:
        selected = self.artifact_table.selection()
        if not selected:
            return
        status = self.artifacts.get(selected[0])
        if status is not None:
            self.artifact_path_var.set(str(status.path))

    def open_selected_artifact(self) -> None:
        selected = self.artifact_table.selection()
        if not selected:
            messagebox.showinfo("Select an artifact", "Select an artifact row first.", parent=self.root)
            return
        status = self.artifacts.get(selected[0])
        if status is None:
            return
        try:
            _open_directory(
                status.path,
                directory_hint=status.kind in ("directory", "collection"),
            )
        except Exception as error:
            messagebox.showerror("Could not open location", str(error), parent=self.root)

    def open_session_log(self) -> None:
        if self.last_log_path is None:
            return
        try:
            if os.name == "nt":
                os.startfile(str(self.last_log_path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(("xdg-open", str(self.last_log_path)), start_new_session=True)
        except Exception as error:
            messagebox.showerror("Could not open session log", str(error), parent=self.root)

    def close(self) -> None:
        active = [item for item in self.running.values() if item.process.poll() is None]
        if active and not messagebox.askyesno(
            "Close research console",
            f"{len(active)} research tool(s) are still running. Close only the console and leave them running?",
            parent=self.root,
        ):
            return
        for running in self.running.values():
            if not running.log_stream.closed:
                running.log_stream.close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ResearchConsole(root)
    root.mainloop()


if __name__ == "__main__":
    main()
