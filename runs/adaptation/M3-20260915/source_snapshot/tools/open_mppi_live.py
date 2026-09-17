"""Open a saved MPPI run in the full native UI without starting optimization."""
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication
from simulator.gui.pva_main_window import PVAResearchWindow

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--follow-sequence',type=Path,help='Follow active_job in a finite prepared trial sequence status.json')
    args=parser.parse_args()
    app=QApplication(sys.argv[:1]);app.setStyle('Fusion');app.setFont(QFont('Segoe UI',10));window=PVAResearchWindow(Path(__file__).resolve().parents[1]);window.show()
    def open_live(job=args.job):
        page=window.mppi_page;page.current_run=job.resolve();page.poll()
        window.main_tabs.setCurrentWidget(page);page.tabs.setCurrentWidget(page.live_view)
    followed=[str(args.job.resolve())]
    def follow_sequence():
        try:job=json.loads(args.follow_sequence.read_text()).get('active_job')
        except (OSError,ValueError):return
        if job and job!=followed[0]:followed[0]=job;open_live(Path(job))
    if args.follow_sequence:
        timer=QTimer(window);timer.setInterval(1500);timer.timeout.connect(follow_sequence);timer.start()
    QTimer.singleShot(400,open_live);sys.exit(app.exec())
