"""Editable CEM task and objective, separate from PPO configuration."""
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QDoubleSpinBox,
    QPushButton,QGroupBox,QScrollArea,QCheckBox)
from planning.cem_objective import REWARD_DEFAULTS,resolve_reward
from .research_widgets import note


class CEMSettingsPage(QWidget):
    save_requested=Signal()
    load_requested=Signal()
    changed=Signal()

    def __init__(self):
        super().__init__();layout=QVBoxLayout(self)
        layout.addWidget(note('Settings apply to the next CEM run. Save to keep them after restarting the app. Each run freezes its own reward, hit criteria and search settings; PPO and existing results keep theirs.'))
        buttons=QHBoxLayout()
        self.save=QPushButton('Save CEM settings');self.save.clicked.connect(self.save_requested.emit);buttons.addWidget(self.save)
        self.load=QPushButton('Load settings from selected CEM run');self.load.clicked.connect(self.load_requested.emit);buttons.addWidget(self.load)
        reset=QPushButton('Reset reward defaults');reset.clicked.connect(lambda:self.set_values({'reward':REWARD_DEFAULTS}));buttons.addWidget(reset);layout.addLayout(buttons)
        scroll=QScrollArea();scroll.setWidgetResizable(True);layout.addWidget(scroll,1)
        body=QWidget();columns=QHBoxLayout(body);scroll.setWidget(body)
        group=QGroupBox('Reward weights and shaping');form=QFormLayout(group);columns.addWidget(group)
        self.reward_spins={}
        fields=[('success_bonus','Valid-hit bonus',2,'Reward once for a valid tip strike.'),
            ('distance_weight','Miss distance cost / m',2,'Cost for the closest tip distance up to the scored terminal event.'),
            ('speed_weight','Directed tip-speed bonus',2,'Bonus scales from zero to this value at the required directed speed, at the closest scored sample.'),
            ('time_weight_per_s','Time cost / s',2,'Cost until the first valid hit, otherwise the proposed duration. Higher values prefer earlier hits.'),
            ('jerk_weight','Squared jerk cost',6,'Cost for mean squared acceleration change per second over the proposed spline. Higher values favor smooth commands.'),
            ('invalid_contact_penalty','Invalid first-contact cost',2,'Cost if a non-tip marker hits first or the first tip entry fails the speed/direction gate.'),
            ('height_weight','Excess height cost / m',2,'Soft cost above the preferred height. The separate maximum-height limit still rejects candidates.'),
            ('preferred_height_m','Preferred peak height [m]',3,'Soft height threshold for predicted drone/cable, independent of the hard maximum.'),
            ('distance_cap_m','Distance cost clip [m]',3,'Maximum distance used by the distance cost.'),
            ('jerk_cap','Squared jerk clip',2,'Cap on the mean squared jerk component before multiplying by its cost weight.')]
        for key,label,decimals,tip in fields:
            spin=QDoubleSpinBox();spin.setDecimals(decimals);spin.setRange(0,1e8);spin.setValue(REWARD_DEFAULTS[key])
            spin.setSingleStep(.001 if key=='jerk_weight' else .01 if key.endswith('_m') else 1)
            spin.setToolTip(tip);form.addRow(label,spin);self.reward_spins[key]=spin;spin.valueChanged.connect(self.changed.emit)
        form.addRow(note('Weights are nonnegative. Set a weight to zero to disable that term. Rewards cannot override an infeasible trajectory.'))
        group=QGroupBox('What counts as a hit');form=QFormLayout(group);columns.addWidget(group)
        self.hit_spins={}
        for key,label,maximum,value in [
            ('tip_target_distance_m','Target radius [m]',10,.05),
            ('minimum_directed_tip_speed_m_s','Minimum directed tip speed [m/s]',100,4),
            ('maximum_tip_velocity_to_desired_direction_error_deg','Maximum direction error [°]',180,45)]:
            spin=QDoubleSpinBox();spin.setDecimals(3);spin.setRange(0,maximum);spin.setValue(value)
            spin.setSingleStep(.01 if key=='tip_target_distance_m' else 1);form.addRow(label,spin)
            self.hit_spins[key]=spin;spin.valueChanged.connect(self.changed.emit)
        self.first_contact=QCheckBox('Only the first tip contact may qualify');self.first_contact.setChecked(True)
        self.tip_first=QCheckBox('Tip must enter before other markers');self.tip_first.setChecked(True)
        for widget in (self.first_contact,self.tip_first):form.addRow(widget);widget.toggled.connect(self.changed.emit)
        self.direction_spins=[]
        for j,axis in enumerate('XYZ'):
            spin=QDoubleSpinBox();spin.setRange(-100,100);spin.setDecimals(4);spin.setSingleStep(.1);spin.setValue(1 if j==0 else 0)
            form.addRow(f'Desired strike direction {axis}',spin);self.direction_spins.append(spin);spin.valueChanged.connect(self.changed.emit)
        form.addRow(note('Direction is a world-frame vector and is normalized when the run starts. It must not be zero. The target position and launch position are on Optimize spline.'))
        form.addRow(note('Batch size, iterations, elite fraction, spline control points, exploration, duration and height limits are also on Optimize spline. Save CEM settings saves those values together with this page.'))
        self.status=note('Ready');layout.addWidget(self.status)

    def values(self):
        return dict(reward=resolve_reward({k:s.value() for k,s in self.reward_spins.items()}),
            success={**{k:s.value() for k,s in self.hit_spins.items()},'first_contact_only':self.first_contact.isChecked(),
                     'tip_must_enter_before_other_markers':self.tip_first.isChecked()},
            desired_strike_direction_world=[s.value() for s in self.direction_spins])

    def set_values(self,settings):
        if 'reward' in settings:
            for key,value in resolve_reward(settings['reward']).items():self.reward_spins[key].setValue(value)
        if 'success' in settings:
            gate=settings['success']
            for key,spin in self.hit_spins.items():
                if key in gate:spin.setValue(gate[key])
            for key,widget in [('first_contact_only',self.first_contact),('tip_must_enter_before_other_markers',self.tip_first)]:
                if key in gate:widget.setChecked(gate[key])
        if 'desired_strike_direction_world' in settings:
            for spin,value in zip(self.direction_spins,settings['desired_strike_direction_world']):spin.setValue(value)
