"""A compact study overview; frozen results, no automatic compute or selection."""
from pathlib import Path
import numpy as np
from PySide6.QtCore import Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QComboBox,
    QScrollArea,QFrame,QSizePolicy)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from experimental_data.system_comparison import load_review, eligible_takes, equal_take_mean
from .model_evolution_page import table, fill
from .research_widgets import note

COLORS=['#2563eb','#0e907d','#b36b20','#8862b5']


def number(value,scale=1.,digits=2):
    return '—' if value is None or not np.isfinite(value) else f'{scale*value:.{digits}f}'


class SystemComparisonPage(QWidget):
    flight_requested=Signal(str,str)
    details_requested=Signal()

    def __init__(self,root,parent=None):
        super().__init__(parent);self.root=Path(root);self.report=None;self.replay_rows=[]
        outer=QVBoxLayout(self);outer.setContentsMargins(0,0,0,0)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setFrameShape(QFrame.Shape.NoFrame);outer.addWidget(scroll)
        content=QWidget();body=QVBoxLayout(content);body.setSpacing(10);scroll.setWidget(content)
        row=QHBoxLayout();body.addLayout(row)
        title=QLabel('M0 → M1 → M2');title.setStyleSheet('font-size:23px;font-weight:650;');row.addWidget(title);row.addStretch()
        refresh=QPushButton('Refresh evidence');refresh.clicked.connect(self.refresh);row.addWidget(refresh)
        doc=QPushButton('Detailed review');doc.clicked.connect(self.open_document);row.addWidget(doc)
        self.status=note('');body.addWidget(self.status)
        self.card_layout=QHBoxLayout();body.addLayout(self.card_layout);self.cards=[]
        row=QHBoxLayout();body.addLayout(row)
        self.mode=QComboBox();self.mode.addItem('Real flights · original forecasts','flights');self.mode.addItem('Same commands · compare models','paired');row.addWidget(self.mode)
        self.dataset=QComboBox();self.dataset.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Fixed);row.addWidget(self.dataset,1)
        self.selection=QComboBox();self.selection.addItem('Excluded from all model fits','excluded');self.selection.addItem('Reserved validation takes','validation');self.selection.addItem('All takes · includes training','all');row.addWidget(self.selection)
        self.context=note('');body.addWidget(self.context)
        self.figure=Figure(figsize=(10,3.4),layout='constrained');self.canvas=FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(280);self.canvas.setMaximumHeight(350);body.addWidget(self.canvas)
        self.take_table=table(['Generation / take','Data use','Drone RMS [cm]','Tip RMS [cm]','Tip coverage','Target distance [cm]'])
        self.take_table.setMinimumHeight(135);self.take_table.setMaximumHeight(170);body.addWidget(self.take_table)
        row=QHBoxLayout();body.addLayout(row)
        self.replay=QPushButton('Replay selected take + original ghost');self.replay.clicked.connect(self.open_replay);row.addWidget(self.replay)
        detail=QPushButton('Model variants, traces & fitting');detail.clicked.connect(self.details_requested);row.addWidget(detail);row.addStretch()
        self.footnote=note('');body.addWidget(self.footnote);body.addStretch()
        self.mode.currentIndexChanged.connect(self.draw);self.dataset.currentIndexChanged.connect(self.draw);self.selection.currentIndexChanged.connect(self.draw)
        self.refresh()

    def open_document(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root/'docs/M0_M1_M2_SYSTEM_COMPARISON.md')))

    def refresh(self):
        self.report=None
        while self.card_layout.count():
            item=self.card_layout.takeAt(0)
            if item.widget():item.widget().deleteLater()
        self.cards=[];self.dataset.blockSignals(True);selected=self.dataset.currentText();self.dataset.clear()
        try:
            self.report=load_review(self.root)
            if self.report is None:
                self.status.setText('No reviewed system comparison yet. Recorded flights and model variants remain available in the other tabs.')
            else:
                r=self.report;n=sum(len(f['takes']) for f in r['flights'].values())
                self.status.setText(f'Development evidence · {len(r["models"])} flown generations · {n} takes · original forecasts preserved')
                for i,m in enumerate(r['models']):
                    f=r['flights'][m['id']];card=QFrame();card.setObjectName('metricCard');layout=QVBoxLayout(card)
                    title=QLabel(f'{m["label"]}   '+['Initial model','First adaptation','Second adaptation'][min(i,2)])
                    title.setStyleSheet(f'color:{COLORS[i%len(COLORS)]};font-size:16px;font-weight:650;');layout.addWidget(title)
                    label=QLabel(f'{f["observed_entries"]} / {len(f["takes"])} virtual target entries')
                    label.setStyleSheet('font-size:17px;font-weight:600;');layout.addWidget(label)
                    text=note(f'Original tip RMS {number(f["mean_tip_rms_m"],100)} cm · drone {number(f["mean_drone_rms_m"],100)} cm')
                    layout.addWidget(text);card.setToolTip(m['id']+'\n'+m['signature']+'\n'+m['interpretation']);self.card_layout.addWidget(card,1);self.cards.append(label)
                for label in r['datasets']:self.dataset.addItem(label)
                index=self.dataset.findText(selected);self.dataset.setCurrentIndex(index if index>=0 else self.dataset.count()-1)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            self.report=None;self.status.setText('Evidence unavailable: '+str(exc))
        finally:self.dataset.blockSignals(False)
        self.draw()

    def draw(self):
        self.figure.clear();self.replay_rows=[];rows=[]
        paired=self.mode.currentData()=='paired';self.dataset.setVisible(paired);self.selection.setVisible(paired)
        self.replay.setEnabled(self.report is not None)
        if not self.report:
            fill(self.take_table,[]);self.canvas.draw_idle();self.context.clear();self.footnote.clear();return
        r=self.report;models=r['models'];ids=[m['id'] for m in models]
        axes=self.figure.subplots(1,3 if paired else 4)
        if paired:
            ds=r['datasets'].get(self.dataset.currentText())
            if not ds:return
            takes=eligible_takes(ds,ids,self.selection.currentData())
            metrics=[('drone','Drone RMS [cm]'),('conditional_cable_tip','Cable-tip RMS [cm]\nMeasured attachment'),('command_driven_tip','Tip RMS [cm]\nPredicted attachment')]
            self.context.setText(f'{self.dataset.currentText()} · {len(takes)} matched takes · causal initial state · fixed commands, times and masks across models')
            for ax,(metric,title) in zip(axes,metrics):
                for t in takes:
                    vals=[ds['models'][mid][t]['metrics'][metric]['rmse_m'] for mid in ids]
                    ax.plot(range(len(ids)),[100*v if v is not None else np.nan for v in vals],color='#ccd2dc',lw=1,zorder=1)
                for i,mid in enumerate(ids):
                    vals=[ds['models'][mid][t]['metrics'][metric]['rmse_m'] for t in takes]
                    self.points(ax,i,vals,100,COLORS[i%len(COLORS)])
                self.axis(ax,title,models)
            for t in takes:
                for m in models:
                    row=ds['models'][m['id']][t];scores=row['metrics'];flown=self.flight_for_take(t)
                    rows.append([m['label']+' / '+t[-3:],row['data_use'],number(scores['drone']['rmse_m'],100),number(scores['command_driven_tip']['rmse_m'],100),number(scores['command_driven_tip']['coverage'],100,1)+'%','—'])
                    self.replay_rows.append(flown)
            self.footnote.setText('Lines pair one measured take; thick marks are equal take means. Cable with measured attachment isolates cable response. Tip with predicted attachment tests the complete command-to-tip chain. Replays always use the original flown ghost. '+(' · '.join(ds['excluded'])))
        else:
            self.context.setText('Each session uses its own flown commands and original forecast. Different rewards and motions: these columns describe the whole system, not model-only improvement.')
            metrics=[('drone','Original drone RMS [cm]',100),('tip','Original tip RMS [cm]',100),('nearest','Nearest target [cm]',100),('speed','Forward speed near target [m/s]',1)]
            for ax,(metric,title,scale) in zip(axes,metrics):
                for i,m in enumerate(models):
                    ts=r['flights'][m['id']]['takes'].values()
                    vals=[v[metric]['rmse_m'] if metric in ('drone','tip') else v['encounter']['nearest'].get('distance_m' if metric=='nearest' else 'outward_speed_m_s') for v in ts]
                    self.points(ax,i,vals,scale,COLORS[i%len(COLORS)])
                if metric=='nearest':ax.axhline(5,color='#8b3e37',ls='--',lw=1);ax.text(.03,.02,'5 cm sphere radius',transform=ax.transAxes,fontsize=8,color='#8b3e37')
                self.axis(ax,title,models)
            for m in models:
                f=r['flights'][m['id']]
                for t,v in f['takes'].items():
                    rows.append([m['label']+' / '+t[-3:],v['role'],number(v['drone']['rmse_m'],100),number(v['tip']['rmse_m'],100),number(v['tip']['coverage'],100,1)+'%',number(v['encounter']['nearest']['distance_m'],100)])
                    self.replay_rows.append((f['batch'],t))
            self.footnote.setText('Dots are individual takes; thick marks are equal take means. RMS covers each whip; closest approach includes 0.5 s of recovery. Clocks are estimated. Near-target speed on a miss is not impact power. Gaps are retained; virtual entries are not measured physical collisions.')
        fill(self.take_table,rows)
        if rows:self.take_table.selectRow(0)
        self.replay.setEnabled(bool(rows));self.canvas.draw_idle()

    def points(self,ax,index,values,scale,color):
        valid=[v for v in values if v is not None and np.isfinite(v)]
        if not valid:return
        ax.scatter(index+np.linspace(-.12,.12,len(valid)),np.array(valid)*scale,color=color,s=29,zorder=3)
        average=equal_take_mean(valid)*scale;ax.plot([index-.23,index+.23],[average,average],color=color,lw=3,zorder=4)
        ax.annotate(number(average),xy=(index,average),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9,color=color)

    def axis(self,ax,title,models):
        # Wrap long titles explicitly for the minimum supported window width.
        ax.set_title(title.replace('Forward speed near target','Near-target forward speed\n'),fontsize=10,pad=14)
        ax.set_xticks(range(len(models)),[m['label'] for m in models]);ax.set_xlim(-.5,len(models)-.5);ax.set_ylim(bottom=0)
        ax.margins(y=.2);ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=9)

    def flight_for_take(self,take):
        for f in self.report['flights'].values():
            if take in f['takes']:return f['batch'],take
        return None

    def open_replay(self):
        index=self.take_table.currentRow()
        if 0<=index<len(self.replay_rows) and self.replay_rows[index]:self.flight_requested.emit(*self.replay_rows[index])
