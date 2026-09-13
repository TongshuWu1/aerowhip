from pathlib import Path
import shutil,math
root=Path(__file__).resolve().parent
live=Path(r'C:/Users/wts28/Lehigh University Dropbox/TonyLehigh Wu/Apps/Overleaf/AeroWhip-ICRA')
files=['main.tex','references.bib','ieeeconf.cls','ieeetrans.cls','figures/aerowhip_tikz_styles.tex','figures/problem_whipping_motion.tex','figures/problem_dder_model.tex']
for sub in ['before','qa']:
    for name in files:
        p=root/sub/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(live/name,p)
preview=Path('output/pdf/AeroWhip_framework_revision_20260913.pdf')
if preview.exists():shutil.copy2(preview,root/'before/preview.pdf')
# Deliberately illustrative coordinates. No measurements or model forecasts.
def curve(t,amount):
    # Shared nominal arch. Deviations vanish at the common initial point.
    x=.30+2.62*t
    y=.36+1.62*math.sin(math.pi*.88*t)+.22*t
    return x-.12*amount*t*t,y-amount*(.15*t+.72*t*t)
def xy(p):return f'({p[0]:.5f},{p[1]:.5f})'
lines=[r'% Schematic local command correction; not simulated or measured data.',
 r'% Curves are tip paths, not cable centerlines. Equal parameters mark equal times.',
 r'\begin{tikzpicture}[x=1cm,y=1cm,',
 r'  every node/.style={font=\fontsize{8}{9.2}\selectfont},',
 r'  refpath/.style={draw=black!65,dashed,line width=.85pt,line cap=round},',
 r'  predpath/.style={draw=awCable,line width=1.25pt,line cap=round},',
 r'  errpath/.style={draw=awAccent,line width=.75pt},',
 r'  sample/.style={circle,inner sep=1.35pt,line width=.65pt,fill=white}]',
 r'\path[use as bounding box] (0,-.13) rectangle (8.35,5.0);',
 r'\node[font=\fontsize{8}{9.2}\selectfont\bfseries] at (1.80,4.78) {(a) Before correction};',
 r'\node[font=\fontsize{8}{9.2}\selectfont\bfseries] at (6.25,4.78) {(b) After correction};',
 r'\node[draw=black!30,rounded corners=1pt,text width=2.65cm,align=center,minimum height=.50cm] at (1.80,4.20) {Latest executed\\command};',
 r'\node[draw=black!30,rounded corners=1pt,text width=2.65cm,align=center,minimum height=.50cm] at (6.25,4.20) {Corrected\\command};',
 r'\draw[awVector] (3.33,4.20)--(4.72,4.20);',
 r'\node[font=\fontsize{7.5}{8.5}\selectfont,anchor=south] at (4.02,4.29) {local update};',
 r'\draw[black!55,line width=.55pt,-{Stealth[length=1.6mm]}] (1.80,3.94)--(1.80,3.61);',
 r'\draw[black!55,line width=.55pt,-{Stealth[length=1.6mm]}] (6.25,3.94)--(6.25,3.61);',
 r'\node[fill=black!4,draw=black!20,rounded corners=1pt,minimum width=7.4cm,minimum height=.44cm] at (4.05,3.36) {Same refined vehicle and cable model (fixed)};',
 r'\draw[black!55,line width=.55pt,-{Stealth[length=1.6mm]}] (1.80,3.11)--(1.80,2.94);',
 r'\draw[black!55,line width=.55pt,-{Stealth[length=1.6mm]}] (6.25,3.11)--(6.25,2.94);',
 r'\node[font=\fontsize{7.5}{8.5}\selectfont,fill=white,inner sep=1pt] at (4.04,2.74) {Tip paths: schematic spatial view};']
for shift,amount in [(.15,1.),(4.6,.16)]:
    lines.append(f'\\begin{{scope}}[shift={{({shift},.30)}}]')
    lines += [r'\draw[black!30,line width=.45pt,-{Stealth[length=1.4mm]}] (.05,.12)--(3.3,.12);',
              r'\draw[black!30,line width=.45pt,-{Stealth[length=1.4mm]}] (.05,.12)--(.05,2.26);',
              r'\node[anchor=north,inner sep=1pt] at (3.22,.09) {$\mathbf e$};']
    for t in [.30,.58,.85]:
        lines.append(r'\draw[errpath] '+xy(curve(t,0))+'--'+xy(curve(t,amount))+';')
    for style,a in [('refpath',0),('predpath',amount)]:
        lines.append('\\draw['+style+'] '+'--'.join(xy(curve(i/90,a)) for i in range(91))+';')
    for t in [.30,.58,.85]:
        lines.append(r'\node[sample,draw=black!65] at '+xy(curve(t,0))+' {};')
        lines.append(r'\node[sample,draw=awCable,fill=awCable] at '+xy(curve(t,amount))+' {};')
    lines += [r'\fill[black] '+xy(curve(0,0))+r' circle (.025);',r'\end{scope}']
lines += [r'\draw[refpath] (.32,-.04)--(.79,-.04);',
 r'\node[anchor=west] at (.85,-.04) {Fixed reference};',
 r'\draw[predpath] (3.13,-.04)--(3.60,-.04);',
 r'\node[anchor=west] at (3.66,-.04) {Prediction};',
 r'\draw[errpath] (5.68,-.04)--(6.10,-.04);',
 r'\node[anchor=west] at (6.16,-.04) {Same-time error};',
 r'\end{tikzpicture}']
figure='\n'.join(lines)+'\n'
(root/'qa/figures/command_correction.tex').write_text(figure,encoding='utf-8')
s=(root/'qa/main.tex').read_text(encoding='utf-8')
anchor=r'\noindent\emph{Command correction with the refined model.}'
block=r'''\begin{figure}[t]
    \centering
    \resizebox{\columnwidth}{!}{\input{figures/command_correction.tex}}
    \caption{Local command correction (schematic).
    The refined model is fixed while commands change toward the
    original tip-motion reference. Paired markers denote matching
    times; red segments show position errors.}
    \label{fig:command_correction}
\end{figure}

'''
assert s.count(anchor)==1;s=s.replace(anchor,block+anchor)
old='''We use the updated predictor to correct commands toward the
initial MPPI plan. Its predicted tip and vehicle trajectories,'''
new=r'''Figure~\ref{fig:command_correction} illustrates local correction
under a fixed refined model: commands change while the desired
tip trajectory and timing remain unchanged. The initial MPPI
plan's predicted tip and vehicle trajectories,'''
assert s.count(old)==1;s=s.replace(old,new)
(root/'qa/main.tex').write_text(s,encoding='utf-8')
(root/'qa/qa.tex').write_text('\\RequirePackage[OT1]{fontenc}\n\\input{main.tex}\n',encoding='utf-8')
# Standalone figure used only for visual QA and PNG export.
(root/'qa/figure_preview.tex').write_text(r'''\documentclass[border=3pt]{standalone}
\RequirePackage[OT1]{fontenc}
\usepackage{amsmath,amssymb,tikz}
\input{figures/aerowhip_tikz_styles.tex}
\begin{document}
\input{figures/command_correction.tex}
\end{document}
''',encoding='utf-8')
print('Staged TikZ schematic and manuscript integration.')
