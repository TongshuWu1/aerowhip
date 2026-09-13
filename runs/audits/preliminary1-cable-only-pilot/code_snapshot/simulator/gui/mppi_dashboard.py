"""Read-only MPPI telemetry; maneuver time is never an optimization deadline."""
import math,json
from PySide6.QtWidgets import QWidget,QVBoxLayout,QGridLayout,QLabel,QFrame,QProgressBar


def readable_log(text):
    lines=[]
    for line in text.splitlines():
        try:row=json.loads(line)
        except (ValueError,TypeError):lines.append(line);continue
        if not isinstance(row,dict) or 'stage' not in row:lines.append(line);continue
        parts=[str(row['stage'])]
        for key,label in [('iteration','total iteration'),('window_iteration','lookahead iteration'),('command_step','command')]:
            if key in row:parts.append(f'{label} {row[key]}')
        if 'maneuver_time_s' in row:parts.append(f'maneuver {row["maneuver_time_s"]:.3f} s')
        if 'error' in row:parts.append(str(row['error']))
        lines.append(' · '.join(parts))
    return '\n'.join(lines)


def snapshot(status,settings,history,windows):
    s=settings.get('mppi',{});receding=s.get('mode')=='receding'
    latest=history[-1] if history else {};committed=windows[-1] if windows else {}
    step=max(status.get('command_step',status.get('command_steps',0)),committed.get('command_step',0))
    current=[r for r in history if r.get('command_step',0)==step] if receding else history
    metrics=current[-1] if current else {}
    anchor=-math.inf;improved_at=0
    for row in current:
        value=row.get('lookahead_score',row.get('best_reward'))
        i=row.get('window_iteration',row.get('iteration',0))
        if value is not None and (not math.isfinite(anchor) or value>anchor+max(.1,abs(anchor)*.005)):
            anchor=value;improved_at=i
    done_iteration=current[-1].get('window_iteration',current[-1].get('iteration',0)) if current else 0
    elapsed=status.get('elapsed_s',latest.get('elapsed_s'))
    seconds=None
    if len(history)>1:
        a,b=history[-2:]
        if a.get('elapsed_s') is not None and b.get('elapsed_s') is not None:seconds=b['elapsed_s']-a['elapsed_s']
    return dict(state=status.get('status','waiting'),stage=status.get('stage',''),step=step,
        iteration=status.get('iteration',status.get('iterations',latest.get('iteration',0))),
        window_iteration=status.get('window_iteration',done_iteration),elapsed=elapsed,seconds=seconds,
        minimum=s.get('initial_minimum_iterations',s.get('minimum_iterations')) if step==0 else s.get('minimum_iterations'),
        patience=s.get('initial_patience',s.get('patience')) if step==0 else s.get('patience'),
        stale=done_iteration-improved_at if current else 0,minimum_done=done_iteration,
        duration=settings.get('task',{}).get('duration_s'),maneuver=status.get('maneuver_time_s',status.get('maneuver_duration_s',committed.get('time_s',step/30))),
        distance=metrics.get('predicted_distance_m',metrics.get('best_minimum_tip_distance_m')),
        hit_fraction=metrics.get('success'),failures=metrics.get('failures'),ess=metrics.get('effective_samples'),
        committed_distance=committed.get('actual_minimum_tip_distance_m',status.get('best_minimum_tip_distance_m')),
        wave=metrics.get('best_candidate_wave_stages'),receding=receding)


class MPPIDashboard(QWidget):
    def __init__(self):
        super().__init__();self.setMinimumHeight(340);body=QVBoxLayout(self);body.setContentsMargins(0,0,0,0)
        self.heading=QLabel('Select a run to see live MPPI statistics');self.heading.setStyleSheet('font-size:15pt;font-weight:650');body.addWidget(self.heading)
        self.context=QLabel();self.context.setWordWrap(True);body.addWidget(self.context)
        grid=QGridLayout();body.addLayout(grid);self.values={}
        for i,(key,label) in enumerate([('iteration','Total iterations'),('elapsed','Recorded elapsed'),('seconds','Last iteration'),('distance','Best lookahead tip distance'),
            ('hit_fraction','Sampled valid hits'),('failures','Sampled infeasible'),('ess','Effective samples'),('step','Committed commands')]):
            card=QFrame();card.setMinimumHeight(70);card.setStyleSheet('QFrame {background:#f4f7fc;border-radius:7px;}');layout=QVBoxLayout(card)
            title=QLabel(label);title.setStyleSheet('color:#526278;font-size:9pt');layout.addWidget(title)
            value=QLabel('—');value.setStyleSheet('font-size:17pt;font-weight:650;color:#172b4d');layout.addWidget(value)
            self.values[key]=value;grid.addWidget(card,i//4,i%4)
        self.maneuver_label=QLabel();body.addWidget(self.maneuver_label);self.maneuver=QProgressBar();self.maneuver.setTextVisible(False);body.addWidget(self.maneuver)
        self.work_label=QLabel();body.addWidget(self.work_label);self.work=QProgressBar();self.work.setTextVisible(False);body.addWidget(self.work)
        self.detail=QLabel();self.detail.setWordWrap(True);body.addWidget(self.detail)

    def update_run(self,status,settings,history,windows,age=None):
        data=snapshot(status,settings,history,windows);s=settings.get('mppi',{})
        headings={'running':'Optimizing','completed':'Finished','stopped':'Stopped','failed':'Failed','prepared':'Ready'}
        self.heading.setText(headings.get(data['state'],'Waiting')+' · '+(data['stage'] or status.get('stop_reason','MPPI')))
        self.context.setText(f'{s.get("horizon_s","—")} s lookahead · {s.get("samples","—")} parallel samples · {settings.get("device","—")} · '+
            ('no iteration limit' if not s.get('iterations') else f'{s["iterations"]} iterations per lookahead'))
        for key,value in self.values.items():
            x=data[key]
            if x is None or not math.isfinite(float(x)):text='—'
            elif key in ('hit_fraction','failures'):text=f'{100*x:.1f}%'
            elif key=='distance':text=f'{100*x:.1f} cm'
            elif key=='elapsed':text=f'{int(x)//60:d}m {int(x)%60:02d}s'
            elif key=='seconds':text=f'{x:.2f} s'
            elif key=='ess':text=f'{x:.1f}'
            else:text=f'{int(x):,}'
            value.setText(text)
        total=data['duration'];time=data['maneuver']
        self.maneuver.setRange(0,1000);self.maneuver.setValue(min(1000,round(1000*time/total)) if total else 0)
        self.maneuver.setVisible(data['receding']);self.maneuver_label.setVisible(data['receding'])
        self.maneuver_label.setText(f'Simulated maneuver: {time:.3f} / {total or 0:g} s safety limit · not a planning-time cap')
        self.work.setRange(0,0) if data['state']=='running' else self.work.setRange(0,1)
        if data['state']!='running':self.work.setValue(1 if data['state']=='completed' else 0)
        self.work_label.setText(f'Lookahead iteration {data["window_iteration"]} · completed {data["minimum_done"]} / {data["minimum"] or "—"} minimum · plateau {data["stale"]} / {data["patience"] or "—"}')
        actual='—' if data['committed_distance'] is None else f'{100*data["committed_distance"]:.1f} cm'
        text=f'Committed path closest tip: {actual}. Sampled hits and lookahead distances describe proposals, not the committed result.'
        if age is not None and data['state']=='running':text+=f' Last status/history write {age:.0f} s ago.'
        if status.get('error'):text+=' '+status['error']
        self.detail.setText(text)
