"""Transport-neutral complete-reference reader/player. No ROS, arming or landing.

CLI validates a bundle only: python fullstate_playback.py /path/to/bundle
"""
import csv
import hashlib
import json
import math
from pathlib import Path


def load_reference(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'fullstate.json').read_text(encoding='utf-8'))
    if metadata.get('schema') not in ('complete_fullstate_reference_v2','recorded_rehearsal_fullstate_v3',
                                    'gentle_recovery_fullstate_v4'):
        raise ValueError('Expected a complete whip + recovery export')
    path = directory/'fullstate_30hz.csv'
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['csv_sha256']:
        raise ValueError('CSV hash mismatch: trajectory is changed or incomplete')
    with path.open(newline='',encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != metadata['sample_count'] or len(rows)<2:
        raise ValueError('Incomplete reference sample count')
    numbers = ('time_s','px_m','py_m','pz_m','vx_m_s','vy_m_s','vz_m_s',
               'ax_m_s2','ay_m_s2','az_m_s2','yaw_rad','yaw_rate_rad_s')
    previous = -1.
    for i,row in enumerate(rows):
        if int(row['sample_index']) != i or row['phase'] not in metadata['phases']:
            raise ValueError('Invalid sample index or phase')
        row['sample_index'] = i
        for key in numbers:
            row[key] = float(row[key])
            if not math.isfinite(row[key]):
                raise ValueError('Nonfinite PVA reference')
        if row['time_s'] <= previous:
            raise ValueError('Nonincreasing reference times')
        previous = row['time_s']
    if rows[0]['time_s'] != 0 or abs(rows[-1]['time_s']-metadata['total_duration_s'])>1e-9:
        raise ValueError('Missing reference start or terminal endpoint')
    if metadata['schema']=='recorded_rehearsal_fullstate_v3':
        expected = metadata['final_position_m']+metadata['final_velocity_m_s']+metadata['final_acceleration_m_s2']
        if any(abs(rows[-1][k]-x)>1e-9 for k,x in zip(numbers[1:10],expected)):
            raise ValueError('Final sample differs from the recorded rehearsal endpoint')
    else:
        if any(abs(rows[-1][k])>1e-9 for k in numbers[4:10]):
            raise ValueError('Reference does not end at zero velocity and acceleration')
        if any(abs(rows[-1][k]-x)>1e-9 for k,x in zip(numbers[1:4],metadata['recovery']['hover_position_m'])):
            raise ValueError('Reference does not finish at the selected hover position')
    return rows, metadata


def play_reference(rows, send_sample, now, sleep, *, cancel_requested=lambda: False,
                   max_lateness_s=.1, on_send=None):
    """Send every row on absolute deadlines; return immediately after the last row.

    Callbacks are supplied by the vehicle integration. Cancellation/timing errors
    raise to that integration's recovery handler; this function never commands a vehicle itself.
    """
    if not rows or not math.isfinite(max_lateness_s) or max_lateness_s<=0:
        raise ValueError('Need reference rows and a finite positive lateness limit')
    start = now()
    for row in rows:
        deadline = start+row['time_s']
        while True:
            if cancel_requested():
                raise InterruptedError('Reference playback cancelled')
            remaining = deadline-now()
            if remaining<=0:
                break
            sleep(min(.001,remaining))
        sent = now()
        if sent-deadline>max_lateness_s:
            raise RuntimeError('Reference deadline missed; stopped before sending stale samples')
        send_sample(row)
        if on_send is not None:
            on_send(dict(sample_index=row['sample_index'],phase=row['phase'],
                         planned_time_s=row['time_s'],send_time_s=sent-start,
                         lateness_s=sent-deadline))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate a complete full-state export; no vehicle connection')
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    rows,meta = load_reference(args.directory)
    print(f'Validated {len(rows)} rows; whip ends {meta["whip_end_s"]:.3f} s; '
          f'complete reference ends {meta["total_duration_s"]:.3f} s.')
