"""Independent MPPI task reward controls."""
from PySide6.QtWidgets import QGroupBox,QFormLayout,QDoubleSpinBox,QPushButton
from .cem_settings_page import CEMSettingsPage
from .reward_page import WEIGHT_FIELDS,SCALE_FIELDS
from .research_widgets import note


class ForceRewardSettings(CEMSettingsPage):
    def __init__(self,reward):
        super().__init__('MPPI')
        old=next(g for g in self.findChildren(QGroupBox) if g.title()=='Reward weights and shaping')
        old.hide()
        group=QGroupBox('MPPI whip objective');form=QFormLayout(group)
        old.parentWidget().layout().insertWidget(0,group)
        form.addRow(note('These MPPI weights score predicted cable strikes. Stop at the first valid hit or the horizon. Recovery is excluded.'))
        self.force_reward_spins={}
        for key,label,unit,default in WEIGHT_FIELDS+SCALE_FIELDS:
            if key not in reward:continue
            spin=QDoubleSpinBox();spin.setDecimals(5);spin.setRange(0,100000);spin.setValue(float(reward[key]))
            self.force_reward_spins[key]=spin;form.addRow(label+unit,spin)
        for button in self.findChildren(QPushButton):
            if button.text()=='Reset reward defaults':button.hide()

    def values(self):
        values=super().values();values.pop('reward')
        values['reward']={k:s.value() for k,s in self.force_reward_spins.items()}
        return values

    def set_values(self,settings):
        super().set_values({k:v for k,v in settings.items() if k!='reward'})
        if hasattr(self,'force_reward_spins'):
            for key,value in settings.get('reward',{}).items():
                if key in self.force_reward_spins:self.force_reward_spins[key].setValue(value)
