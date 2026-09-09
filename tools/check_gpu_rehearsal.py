"""Run a native GPU simulation rehearsal, inspect its CSV, execute once and report timing."""
import argparse
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from simulator.gui.main_window import SimulatorMainWindow
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--target', type=float, nargs=3, default=[1.02,.01,1.4])
    parser.add_argument('--fullstate', action='store_true')
    args = parser.parse_args()
    app = QApplication([])
    window = SimulatorMainWindow(ROOT, *[json.loads((ROOT/'config'/f'{n}.json').read_text())
                                         for n in ('model','task','ppo')])
    window.show()
    window.main_tabs.setCurrentIndex(6 if args.fullstate else 5)
    page = window.fullstate_page if args.fullstate else window.testing_page
    path = str(args.checkpoint.resolve())
    index = page.checkpoints.findData(path)
    if index < 0:
        page.checkpoints.addItem(args.checkpoint.name, path)
        index = page.checkpoints.count()-1
    page.checkpoints.setCurrentIndex(index)
    for spin, value in zip(page.target_spins,args.target):
        spin.setValue(value)
    page.update_target()
    state = dict(executed=False,ticks=0)
    errors = []

    def fail(kind,error,tb):
        traceback.print_exception(kind,error,tb)
        errors.append(str(error))
        page.stop_rehearsal()
    sys.excepthook = fail

    def poll():
        state['ticks'] += 1
        if state['ticks'] % 50 == 0:
            print('STATUS', page.status.text(), flush=True)
        if page.execute.isEnabled() and not state['executed']:
            assert page.table.rowCount() > 0
            print('PLAN', page.plan_note.text(), flush=True)
            state['executed'] = True
            page.execute_sequence()

    def done():
        summary = json.loads((page.directory/'flight.json').read_text())
        print('RESULT', json.dumps(dict(directory=str(page.directory), outcome=summary['outcome'],
            timing=summary.get('timing'), events=summary['events'])), flush=True)
        if not summary['outcome'].startswith('Completed'):
            errors.append(summary['outcome'])
        page.viewer.plotter.screenshot(str(page.directory/'gpu_rehearsal.png'))
        window.close()
        app.exit(1 if errors else 0)

    def start():
        page.start_rehearsal()
        if page.worker is None:
            print(page.status.text(), flush=True)
            window.close()
            app.exit(1)
        else:
            page.flight_finished.connect(done)

    def timeout():
        errors.append('timeout')
        page.stop_rehearsal()

    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(100)
    QTimer.singleShot(300,start)
    QTimer.singleShot(120000,timeout)
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
