"""PPO reward weights for the independent force MPPI workspace."""
from PySide6.QtWidgets import QGroupBox,QFormLayout,QDoubleSpinBox,QPushButton
from .cem_settings_page import CEMSettingsPage
from .reward_page import WEIGHT_FIELDS,SCALE_FIELDS
from .research_widgets import note


class ForceRewardSettings(CEMSettingsPage):
    def __init__(self,reward):
        super().__init__('MPPI')
        old=next(g for g in self.findChildren(QGroupBox) if g.title()=='Reward weights and shaping')
        old.hide()
        group=QGroupBox('PPO execution reward');form=QFormLayout(group)
        old.parentWidget().layout().insertWidget(0,group)
        form.addRow(note('Same reward equations as the selected PPO run. These weights belong only to new MPPI force plans; recovery is excluded.'))
        self.ppo_spins={}
        for key,label,unit,default in WEIGHT_FIELDS+SCALE_FIELDS:
            if key not in reward:continue
            spin=QDoubleSpinBox();spin.setDecimals(5);spin.setRange(0,100000);spin.setValue(float(reward[key]))
            self.ppo_spins[key]=spin;form.addRow(label+unit,spin)
        for button in self.findChildren(QPushButton):
            if button.text()=='Reset reward defaults':button.hide()

    def values(self):
        values=super().values();values.pop('reward')
        values['ppo_reward']={k:s.value() for k,s in self.ppo_spins.items()}
        return values

    def set_values(self,settings):
        super().set_values({k:v for k,v in settings.items() if k!='reward'})
        if hasattr(self,'ppo_spins'):
            for key,value in settings.get('ppo_reward',{}).items():
                if key in self.ppo_spins:self.ppo_spins[key].setValue(value)
