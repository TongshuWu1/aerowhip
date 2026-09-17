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
        minimum=s.get('initial_minimum_iterations',s.get('minimum_iterations')) if receding and step==0 else s.get('minimum_iterations'),
        patience=s.get('initial_patience',s.get('patience')) if receding and step==0 else s.get('patience'),
        stale=done_iteration-improved_at if current else 0,minimum_done=done_iteration,
        duration=settings.get('task',{}).get('duration_s'),maneuver=status.get('maneuver_time_s',status.get('maneuver_duration_s',committed.get('time_s',step/30))),
        distance=metrics.get('strike_distance_m',metrics.get('predicted_distance_m',metrics.get('best_minimum_tip_distance_m'))),
        directed_speed=metrics.get('directed_tip_speed_m_s',status.get('directed_tip_speed_m_s')),
        speed_gain=metrics.get('rewarded_tip_speed_m_s',status.get('rewarded_tip_speed_m_s')),
        hit_fraction=metrics.get('accepted_fraction',metrics.get('success')),failures=(1-metrics['physics_valid_fraction'] if 'physics_valid_fraction' in metrics else metrics.get('failures')),ess=metrics.get('effective_samples'),
        committed_distance=committed.get('actual_minimum_tip_distance_m',status.get('best_minimum_tip_distance_m')),
        wave=metrics.get('best_candidate_wave_stages'),receding=receding)


class MPPIDashboard(QWidget):
    def __init__(self):
        super().__init__();self.setMinimumHeight(340);body=QVBoxLayout(self);body.setContentsMargins(0,0,0,0)
        self.heading=QLabel('Select a run to see live MPPI statistics');self.heading.setStyleSheet('font-size:15pt;font-weight:650');body.addWidget(self.heading)
        self.context=QLabel();self.context.setWordWrap(True);body.addWidget(self.context)
        grid=QGridLayout();body.addLayout(grid);self.values={};self.labels={}
        for i,(key,label) in enumerate([('iteration','Total iterations'),('elapsed','Recorded elapsed'),('seconds','Last iteration'),('distance','Best lookahead tip distance'),
            ('hit_fraction','Sampled valid hits'),('failures','Sampled infeasible'),('ess','Effective samples'),('step','Committed commands')]):
            card=QFrame();card.setMinimumHeight(70);card.setStyleSheet('QFrame {background:#f4f7fc;border-radius:7px;}');layout=QVBoxLayout(card)
            title=QLabel(label);title.setStyleSheet('color:#526278;font-size:9pt');layout.addWidget(title);self.labels[key]=title
            value=QLabel('—');value.setStyleSheet('font-size:17pt;font-weight:650;color:#172b4d');layout.addWidget(value)
            self.values[key]=value;grid.addWidget(card,i//4,i%4)
        self.maneuver_label=QLabel();body.addWidget(self.maneuver_label);self.maneuver=QProgressBar();self.maneuver.setTextVisible(False);body.addWidget(self.maneuver)
        self.work_label=QLabel();body.addWidget(self.work_label);self.work=QProgressBar();self.work.setTextVisible(False);body.addWidget(self.work)
        self.detail=QLabel();self.detail.setWordWrap(True);body.addWidget(self.detail)

    def update_run(self,status,settings,history,windows,age=None):
        data=snapshot(status,settings,history,windows);s=settings.get('mppi',{})
        headings={'running':'Optimizing','completed':'Finished','stopped':'Stopped','failed':'Failed','prepared':'Ready'}
        self.heading.setText(headings.get(data['state'],'Waiting')+' · '+(data['stage'] or status.get('stop_reason','MPPI').replace('_',' ')))
        self.context.setText(f'{s.get("horizon_s","—")} s lookahead · {s.get("samples","—")} parallel samples · {settings.get("device","—")} · '+
            ('no iteration limit' if not s.get('iterations') else f'{s["iterations"]} iterations per lookahead'))
        offline=s.get('mode')=='open_loop'
        if offline:self.context.setText(f'{data["duration"]:g} s complete whip · {s["samples"]} samples · {s["support_points"]} control points · {s["proposal_count"]} proposals · adaptive temperature')
        spline=settings.get('command_contract')=='position_spline_pva_30hz_v1'
        free_target=settings.get('trajectory_objective',{}).get('free_target',False)
        release_metrics=spline or free_target
        gain_mode=settings.get('trajectory_objective',{}).get('speed_metric')=='tip_gain_over_root'
        for key,label in [('distance','Best scored strike distance' if release_metrics else 'Best lookahead tip distance'),
                          ('hit_fraction','Feasible strike candidates' if release_metrics else 'Sampled valid hits'),
                          ('step',('Rewarded tip-speed gain' if gain_mode else 'Forward tip speed at strike') if release_metrics else 'Committed commands')]:self.labels[key].setText(label)
        if release_metrics:
            data['step']=data['speed_gain'] if gain_mode else data['directed_speed']
            data['iteration']=data['minimum_done']
        if free_target:
            self.labels['distance'].setText('Cable height RMS at release')
            data['distance']=(history[-1] if history else status).get('horizontal_cable_rms_m')
        for key,value in self.values.items():
            x=data[key]
            if x is None or not math.isfinite(float(x)):text='—'
            elif key in ('hit_fraction','failures'):text=f'{100*x:.1f}%'
            elif key=='distance':text=f'{100*x:.2f} cm'
            elif key=='step' and release_metrics:text=f'{x:.2f} m/s'
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
        if offline:
            text=('Complete motion optimized from the saved start.' if data['state']=='completed' else 'Optimizing the complete motion from the same start; no commands are committed during search.')
            latest=history[-1] if history else {}
            parts=latest.get('objective_components',{})
            if parts:text+=' Best score contributions: '+', '.join(f'{k} {v:+.1f}' for k,v in parts.items())+'.'
        if age is not None and data['state']=='running':text+=f' Last status/history write {age:.0f} s ago.'
        if status.get('error'):text+=' '+status['error']
        if release_metrics and settings.get('trajectory_objective',{}).get('schema')=='targeted_fold_strike_v1':
            representation='12 spline points / 9 adjustable' if spline else f'{s["support_points"]} jerk controls'
            self.context.setText(f'{data["duration"]:g} s full whip · {representation} · {s["samples"]} candidates · {s["iterations"]} iterations')
            self.work_label.setText(f'Completed iteration {data["iteration"]} / {s["iterations"]}')
            self.work.setRange(0,s['iterations']);self.work.setValue(int(data['iteration']))
            latest=history[-1] if history else {}
            text+=(' Travelling fold is required.' if settings.get('fold_requirement','required')=='required' else ' Fold propagation is diagnostic only.')+' Recovery: brake, return, hold.'
            angle=latest.get('strike_angle_deg');limit=settings.get('trajectory_objective',{}).get('maximum_strike_angle_deg')
            if angle is not None:text+=f' Strike angle: {angle:.1f} deg'+(f' / {limit:g} deg maximum.' if limit is not None else '.')
            if settings.get('trajectory_objective',{}).get('prefer_aligned_strike',False):text+=' Smaller angles receive more speed credit.'
            gain=latest.get('rewarded_tip_speed_m_s');root=latest.get('root_forward_speed_m_s')
            if gain is not None and settings.get('trajectory_objective',{}).get('speed_metric')=='tip_gain_over_root':text+=f' Tip forward speed: {data["directed_speed"]:.2f} m/s; root forward speed: {root:.2f} m/s.'
            minimum_gain=settings.get('trajectory_objective',{}).get('minimum_tip_speed_gain_m_s')
            if minimum_gain is not None:text+=f' Required speed gain: {minimum_gain:g} m/s.'
            if latest.get('best_score') is None:
                text+=' No feasible strike candidate yet.'
                sampled=latest.get('best_sampled_speed_gain_m_s')
                if sampled is not None:text+=f' Best sampled gain: {sampled:.2f} m/s (exploration only).'
        if settings.get('trajectory_objective',{}).get('schema')=='preferred_fold_v1':
            self.labels['distance'].setText('Closest tip distance')
            self.labels['hit_fraction'].setText('Sampled valid tip contacts')
            text+=' Original M0 cable-shape objective; smooth alignment reward, no angle cutoff. Smooth brake, return and hover.'
            self.work_label.setText(f'Completed {data["minimum_done"]} updates; plateau {data["stale"]} / {data["patience"]}')
        self.detail.setText(text)
