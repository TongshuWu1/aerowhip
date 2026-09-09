"""Editable reward settings for the point-force PPO task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from experimental_data.io import atomic_json

from .reward_explanations import FIELD_HELP, HIT_FIELDS


WEIGHT_FIELDS = (
    ("progress_weight", "Approaching the target", "", 20.0),
    (
        "strike_quality_improvement_weight",
        "Tip speed near target",
        "",
        60.0,
    ),
    (
        "point_displacement_integral_weight",
        "Displacement over time",
        " /s",
        0.0,
    ),
    ("success_bonus", "Valid hit bonus", "", 100.0),
    ("success_forward_return_bonus_weight", "Attachment return at hit", "", 50.0),
    ("success_release_bonus_weight", "Release at hit", "", 25.0),
    ("non_tip_first_penalty", "Other marker hits first", "", 25.0),
    ("invalid_tip_entry_penalty", "Invalid tip contact", "", 0.0),
    ("timeout_penalty", "Timeout", "", 0.0),
    ("terminal_displacement_weight", "Displacement at hit", "", 40.0),
    ("maximum_displacement_weight", "Maximum drone travel", "", 0.0),
    ("time_to_success_weight_per_s", "Elapsed time", " /s", 1.0),
    ("numerical_failure_penalty", "Simulation failure", "", 100.0),
)

SCALE_FIELDS = (
    ("proximity_scale_m", "Target proximity width", " m", 0.08),
    ("directed_speed_reward_cap_m_s", "Speed shaping cap", " m/s", 4.0),
    ("relative_directed_speed_reward_cap_m_s", "Combined reward: relative speed cap", " m/s", 6.0),
    ("forward_excursion_scale_m", "Forward excursion scale", " m", 0.35),
    ("point_backward_speed_scale_m_s", "Point return-speed scale", " m/s", 1.0),
    ("relative_tip_forward_speed_scale_m_s", "Relative tip-speed scale", " m/s", 4.0),
    ("displacement_cost_scale_m", "Displacement-cost scale", " m", 0.35),
)


class CompactDoubleSpinBox(QDoubleSpinBox):
    def textFromValue(self, value: float) -> str:  # noqa: N802
        text = super().textFromValue(value)
        return text.rstrip('0').rstrip(self.locale().decimalPoint()) if self.decimals() else text


class RewardSettingsPage(QWidget):
    settings_saved = Signal()

    def __init__(
        self,
        project_root: Path,
        ppo_config: dict[str, Any],
        parent: QWidget | None = None,
        config_directory: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.project_root = project_root
        self.config_directory = Path(config_directory) if config_directory else project_root/'config'
        self.ppo_config = ppo_config
        self.task_config = json.loads((self.config_directory / 'task.json').read_text(encoding="utf-8"))
        self.spins: dict[str, QDoubleSpinBox] = {}
        self.hit_spins: dict[str, QDoubleSpinBox] = {}
        self.task_spins = {}
        self.action_spins = {}
        self._saved_task = {}
        self._saved_reward: dict[str, float] = {}
        self._saved_hit: dict[str, float] = {}
        self._loading = True
        self._build_ui()
        model_path=self.config_directory/'model.json'
        saved_model=json.loads(model_path.read_text(encoding='utf-8')) if model_path.is_file() else {}
        self.native_fullstate = saved_model.get('fullstate_execution',{}).get('schema')=='tracked_pose_execution_v1'
        if self.native_fullstate:
            self.task_spins[('control_dt_s',None)].setEnabled(False)
            self.task_spins[('control_dt_s',None)].setToolTip('Native 30 Hz force / FullState contract. A different clock requires a separately designed model.')
        self._load_values(ppo_config["reward"])

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 18)
        outer.setSpacing(12)
        self.setStyleSheet("QDoubleSpinBox { min-height: 24px; padding: 1px 8px; }")

        toolbar = QHBoxLayout()
        self.status_text = QLabel("Saved")
        self.status_text.setObjectName("mutedText")
        toolbar.addWidget(self.status_text)
        toolbar.addWidget(self._note("Changes apply to the next training run."), 1)
        reload_button = QPushButton("Reload")
        reload_button.clicked.connect(self.reload_saved)
        self.save_button = QPushButton("Apply task and rewards")
        self.save_button.setObjectName("primaryButton")
        self.save_button.clicked.connect(self.save_settings)
        toolbar.addWidget(reload_button)
        toolbar.addWidget(self.save_button)
        outer.addLayout(toolbar)

        scroll = QScrollArea(self)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(16)

        contract = self._note('Plan from the initial state and known target, then execute one frozen sequence. '
                              'Hit conditions score the simulation. Real execution ends at the planned cutoff; it does not wait for hit feedback.')
        layout.addWidget(contract)
        setup = QGroupBox('Strike task')
        columns = QHBoxLayout(setup)
        target_form, limits_form = QFormLayout(), QFormLayout()
        columns.addLayout(target_form, 1)
        columns.addLayout(limits_form, 1)
        for key, title in (('target_position_m', 'Target'),
                           ('desired_strike_direction_world', 'Hit direction'),
                           ('initial_root_position_m', 'Initial attachment')):
            row = QHBoxLayout()
            for axis in range(3):
                spin = self._make_spin(suffix='', minimum=-20, maximum=20)
                spin.setMinimumWidth(65)
                spin.setToolTip(f'{title} {"XYZ"[axis]}' + (' [m]' if key != 'desired_strike_direction_world' else ''))
                self.task_spins[(key, axis)] = spin
                row.addWidget(spin)
            target_form.addRow(title + ' · X / Y / Z', row)
        for key, label, suffix, lower, upper in (
            ('episode_duration_s', 'Maximum plan duration', ' s', .1, 20),
            ('control_dt_s', 'Policy frequency', ' Hz', 1, 150)):
            spin = self._make_spin(suffix=suffix, minimum=lower, maximum=upper)
            self.task_spins[(key, None)] = spin
            limits_form.addRow(label, spin)
        for key, label in (('maximum_force_norm_n', 'Maximum total force'),
                           ('minimum_vertical_force_n', 'Minimum upward force')):
            spin = self._make_spin(suffix=' N', minimum=0, maximum=100)
            self.action_spins[key] = spin
            limits_form.addRow(label, spin)
        layout.addWidget(setup)
        self.execution_note = self._note('')
        layout.addWidget(self.execution_note)

        hit_group = QGroupBox("Hit conditions")
        hit_layout = QVBoxLayout(hit_group)
        hit_fields = QHBoxLayout()
        for key, label, suffix, lower, upper, explanation in HIT_FIELDS:
            field = QVBoxLayout()
            title = QLabel(label)
            title.setToolTip(explanation)
            spin = self._make_spin(suffix=suffix, minimum=lower, maximum=upper)
            spin.setToolTip(explanation)
            self.hit_spins[key] = spin
            field.addWidget(title)
            field.addWidget(spin)
            hit_fields.addLayout(field, 1)
        hit_layout.addLayout(hit_fields)
        self.hit_note = self._note("")
        hit_layout.addWidget(self.hit_note)
        layout.addWidget(hit_group)

        weights = QHBoxLayout()
        weights.setSpacing(16)
        weights.addWidget(self._weight_group("Rewards", (
            "progress_weight", "strike_quality_improvement_weight", "success_bonus",
            "success_forward_return_bonus_weight", "success_release_bonus_weight",
        )), 1)
        weights.addWidget(self._weight_group("Penalties", (
            "time_to_success_weight_per_s", "terminal_displacement_weight",
            "maximum_displacement_weight",
            "point_displacement_integral_weight", "non_tip_first_penalty",
            "invalid_tip_entry_penalty", "timeout_penalty", "numerical_failure_penalty",
        )), 1)
        layout.addLayout(weights)

        self.advanced_button = QPushButton("Show advanced scales")
        self.advanced_button.setCheckable(True)
        self.advanced_panel = QGroupBox("Shaping scales")
        scales_form = QFormLayout(self.advanced_panel)
        self.speed_reference=QComboBox()
        self.speed_reference.addItem('World frame (impact speed)','world')
        self.speed_reference.addItem('Relative to drone','attachment_relative')
        self.speed_reference.addItem('World × relative to attachment','world_and_attachment_relative')
        self.speed_reference.currentIndexChanged.connect(self._refresh_summary)
        scales_form.addRow('Speed reward reference',self.speed_reference)
        self.allowance_mode = QComboBox()
        self.allowance_mode.addItem('Charge all displacement', 'none')
        self.allowance_mode.addItem('Required reach + preparation margin', 'required_reach_plus_margin')
        self.allowance_mode.currentIndexChanged.connect(self._refresh_summary)
        scales_form.addRow('Travel allowance', self.allowance_mode)
        margin = self._make_spin(suffix=' m', minimum=0., maximum=5.)
        self.spins['displacement_allowance_margin_m'] = margin
        self._field(scales_form, 'Preparation margin',
            'No travel cost inside max(0, start-to-target distance − cable length − hit radius) + margin. '
            'Uses each sampled target and initial attachment. This is a soft reward allowance, not a workspace safety limit.', margin)
        for key, label, suffix, _default in SCALE_FIELDS:
            spin = self._make_spin(suffix=suffix, minimum=0.001, maximum=1000.0)
            self._field(scales_form, label, FIELD_HELP[key], spin)
            self.spins[key] = spin
        self.advanced_panel.hide()
        self.advanced_button.toggled.connect(self.advanced_panel.setVisible)
        self.advanced_button.toggled.connect(
            lambda checked: self.advanced_button.setText(
                "Hide advanced scales" if checked else "Show advanced scales"
            )
        )
        layout.addWidget(self.advanced_button, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.advanced_panel)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

    @staticmethod
    def _note(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setObjectName("mutedText")
        return label

    def _field(self, form: QFormLayout, label: str, description: str, spin: QDoubleSpinBox) -> None:
        title = QLabel(label)
        title.setWordWrap(True)
        title.setToolTip(description)
        spin.setToolTip(description)
        spin.setFixedWidth(112)
        form.addRow(title, spin)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

    def _weight_group(self, title: str, keys: tuple[str, ...]) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        form = QFormLayout()
        lookup = {row[0]: row for row in WEIGHT_FIELDS}
        for key in keys:
            _, label, suffix, _default = lookup[key]
            spin = self._make_spin(suffix=suffix, minimum=0, maximum=10000)
            spin.setSpecialValueText("Off")
            self.spins[key] = spin
            self._field(form, label, FIELD_HELP[key], spin)
        layout.addLayout(form)
        layout.addStretch(1)
        return group

    def _make_spin(
        self, *, suffix: str, minimum: float, maximum: float
    ) -> QDoubleSpinBox:
        spin = CompactDoubleSpinBox(self)
        spin.setRange(minimum, maximum)
        spin.setDecimals(4)
        spin.setSingleStep(0.5 if minimum == 0.0 else 0.05)
        spin.setSuffix(suffix)
        spin.valueChanged.connect(self._refresh_summary)
        return spin

    def _load_values(self, reward: dict[str, Any]) -> None:
        reward = {'maximum_displacement_weight':0., 'relative_directed_speed_reward_cap_m_s':6.,
                  'displacement_allowance_margin_m':0., **reward}
        self._loading = True
        deployment = self.ppo_config.get('deployment', {})
        if self.native_fullstate:
            self.execution_note.setText('Native 30 Hz force plan → frozen FullState reference → fitted drone and cable, both residuals. '
                'Preparation is included in the maximum plan duration; gentle recovery is appended only during export. '
                'Combined speed shaping rewards forward world tip motion and forward tip motion relative to the attachment together. '
                'The geometric travel allowance is a lower-bound estimate, not proof that a trajectory is reachable.')
        elif deployment.get('enabled', False):
            self.execution_note.setText(
                f"Saved execution settings: {1000 * deployment.get('strike_followthrough_s', 0.):g} ms fixed follow-through · "
                f"{deployment['recovery_duration_s']:g} s PID recovery allowance · "
                f"−{deployment['recovery_failure_penalty']:g} if recovery does not settle."
            )
        else:
            self.execution_note.setText('')
        self.speed_reference.setCurrentIndex(self.speed_reference.findData(reward.get('directed_speed_shaping_reference','attachment_relative')))
        self.allowance_mode.setCurrentIndex(self.allowance_mode.findData(reward.get('displacement_allowance_mode','none')))
        for key, spin in self.spins.items():
            spin.blockSignals(True)
            spin.setValue(float(reward[key]))
            spin.blockSignals(False)
        for key, spin in self.hit_spins.items():
            spin.setValue(float(self.task_config['success'][key]))
        for (key, axis), spin in self.task_spins.items():
            value = self.task_config[key]
            spin.setValue(1. / float(value) if key == 'control_dt_s' else float(value if axis is None else value[axis]))
        for key, spin in self.action_spins.items():
            spin.setValue(float(self.ppo_config['action'][key]))
        self._saved_task = self._task_values()
        self._saved_reward = self._reward_values()
        self._saved_hit = self._hit_values()
        self._loading = False
        self._refresh_summary()

    def _reward_values(self) -> dict[str, float]:
        return {**{key: spin.value() for key, spin in self.spins.items()},
                'directed_speed_shaping_reference':self.speed_reference.currentData(),
                'displacement_allowance_mode':self.allowance_mode.currentData()}

    def _hit_values(self) -> dict[str, float]:
        return {key: spin.value() for key, spin in self.hit_spins.items()}

    def _task_values(self):
        return {**{str(key): spin.value() for key, spin in self.task_spins.items()},
                **{key: spin.value() for key, spin in self.action_spins.items()}}

    @property
    def has_unsaved_changes(self) -> bool:
        return (self._reward_values() != self._saved_reward or self._hit_values() != self._saved_hit
                or self._task_values() != self._saved_task)

    def _refresh_summary(self) -> None:
        if self._loading:
            return
        self.status_text.setText("Unsaved changes" if self.has_unsaved_changes else "Saved")
        self.save_button.setEnabled(self.has_unsaved_changes)
        order = "Tip must enter first. " if self.task_config["success"]["tip_must_enter_before_other_markers"] else ""
        if self.task_config['success'].get('first_contact_only', False):
            order += 'First contact must be valid; an invalid contact ends the attempt. '
        self.hit_note.setText(order + "Speed and angle use world tip velocity. Angle is a hit condition only.")

    def page_activated(self) -> None:
        if not self.has_unsaved_changes:
            self.reload_saved()

    @staticmethod
    def _equation(values: dict[str, float], *, multiline: bool) -> str:
        separator = "\n" if multiline else " "
        displacement = ('max(0, displacement − required-reach allowance − preparation margin)'
                        if values.get('displacement_allowance_mode') == 'required_reach_plus_margin' else 'point displacement')
        quality = ('proximity × world-speed quality × attachment-relative-speed quality'
                   if values.get('directed_speed_shaping_reference') == 'world_and_attachment_relative'
                   else f"near-target {values.get('directed_speed_shaping_reference','attachment_relative')} target-directed tip-speed quality")
        terms = (
            f"+ {values['progress_weight']:g} × Δ(best normalized tip progress)",
            f"+ {values['strike_quality_improvement_weight']:g} × Δ(best {quality})",
            f"+ {values['success_bonus']:g} × first valid hit",
            f"− {values['point_displacement_integral_weight']:g} × ∫ log1p(({displacement} / {values['displacement_cost_scale_m']:g} m)²) dt",
            f"+ {values['success_forward_return_bonus_weight']:g} × successful return quality",
            f"+ {values['success_release_bonus_weight']:g} × successful release quality",
            f"− {values['non_tip_first_penalty']:g} × non-tip-first",
            f"− {values['invalid_tip_entry_penalty']:g} × invalid tip entry",
            f"− {values['timeout_penalty']:g} × timeout",
            f"− {values['terminal_displacement_weight']:g} × successful terminal displacement cost",
            f"− {values.get('maximum_displacement_weight',0):g} × max log1p(({displacement} / {values['displacement_cost_scale_m']:g} m)²)",
            f"− {values['time_to_success_weight_per_s']:g} × elapsed seconds",
            f"− {values['numerical_failure_penalty']:g} × numerical failure",
            "+ 0 × impact-angle quality",
        )
        return separator.join(terms)

    def reload_saved(self) -> None:
        path = self.config_directory / "ppo.json"
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
            self.task_config = json.loads((self.config_directory / 'task.json').read_text(encoding='utf-8'))
            self.ppo_config.clear()
            self.ppo_config.update(config)
            self._load_values(config["reward"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            self.status_text.setText(f"Could not reload reward settings: {error}")
            return
        self.status_text.setText("Saved")

    def save_settings(self) -> None:
        values = self._reward_values()
        ppo_path = self.config_directory / "ppo.json"
        task_path = self.config_directory / "task.json"
        try:
            # Merge this editor's fields into the latest saved config. Do not
            # overwrite training controls changed elsewhere since the page loaded.
            config = json.loads(ppo_path.read_text(encoding='utf-8'))
            task = json.loads(task_path.read_text(encoding="utf-8"))
            reward = config['reward']
            reward.update(values)
            reward.update(angle_shaping_weight=0.0, angle_is_binary_success_gate_only=True,
                          equation=self._equation(values, multiline=False))
            task['success'].update(self._hit_values())
            for (key, axis), spin in self.task_spins.items():
                if axis is None:
                    task[key] = 1. / spin.value() if key == 'control_dt_s' else spin.value()
                else:
                    task[key][axis] = spin.value()
            config['action'].update({key: spin.value() for key, spin in self.action_spins.items()})
            if sum(value ** 2 for value in task['desired_strike_direction_world']) < 1e-12:
                raise ValueError('The hit direction must be nonzero.')
            from run_ppo import validate_contract
            from simulator.workflow import read_json, stamp
            validate_contract(read_json(self.config_directory / 'model.json'), task, config)
            task.setdefault("reward", {})["angle_shaping_weight"] = 0.0
            task["reward"]["angle_is_binary_success_gate_only"] = True
            atomic_json(ppo_path, config)
            atomic_json(task_path, task)
            version = self.project_root / 'data/task_versions' / stamp()
            atomic_json(version / 'task.json', task)
            atomic_json(version / 'ppo.json', config)
            self.ppo_config.clear()
            self.ppo_config.update(config)
            self.task_config = task
            self._load_values(reward)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            self.status_text.setText(f"Could not save reward settings: {error}")
            return
        self.status_text.setText("Saved")
        self.settings_saved.emit()
