"""Read-only audit: tracked-origin hover height, exact saved forecast, no fit."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from experimental_data.adaptation_check import (
    load_comparison, flight_names, read_optitrack, read_controller, interpolate_positions, sha256)

OUTPUT = Path(__file__).resolve().parent
BATCH = ROOT / 'rehearsal_csv_and_result_in_real_flight/20260908-195207-486249-seed655_best_validation/adp0'


def main():
    rows = []
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for i, take in enumerate(flight_names(BATCH)):
        comparison = load_comparison(ROOT, BATCH, take)
        mocap = read_optitrack(BATCH / 'flight_take' / (take + '.csv'))
        control = read_controller(BATCH / 'flight_take' / ('experiment_' + take + '.csv'))
        time = mocap['time'] + comparison['alignment']['offset_s'] - comparison['onset']
        ct = control['time_s'] - comparison['onset']
        cp = np.column_stack([control[k] for k in ('x', 'y', 'z')])
        command = np.column_stack([control[k] for k in ('cmd_x', 'cmd_y', 'cmd_z', 'cmd_vx', 'cmd_vy', 'cmd_vz', 'cmd_ax', 'cmd_ay', 'cmd_az')])
        row = dict(take=take, hashes=comparison['hashes'], alignment=comparison['alignment'], phases={})
        for j, (name, start, end) in enumerate([('before_whip', -.21, 0.), ('final_hold', 9.2, 10.9)]):
            mask = (time >= start) & (time < end)
            observed = mocap['drone'][mask]
            controller_mask = (ct >= start) & (ct < end)
            assert observed.shape[0] >= 20
            np.testing.assert_allclose(command[controller_mask], np.broadcast_to([-2, 0, 1.255, 0, 0, 0, 0, 0, 0], command[controller_mask].shape), atol=1e-12, rtol=0)
            assert np.all(control['cmd_valid'][controller_mask] == 1)
            controller = interpolate_positions(ct, cp, time[mask])
            if name == 'before_whip':
                prediction_z = comparison['metadata']['initial_tracking_origin_m'][2]
                prediction_kind = 'saved nominal start; no pre-onset prediction exists'
            else:
                prediction_z = float(np.mean(comparison['predicted_origin'][(comparison['time'] >= start) & (comparison['time'] < end), 2]))
                prediction_kind = 'mean exact saved preflight forecast in same interval'
            delta_stream = observed - controller
            values = dict(window_s=[start, end], samples=len(observed), command_xyz_m=[-2., 0., 1.255],
                          measured_mean_xyz_m=observed.mean(0).tolist(), measured_std_xyz_m=observed.std(0).tolist(),
                          saved_prediction_mean_z_m=prediction_z, prediction_kind=prediction_kind,
                          measured_minus_command_z_cm=float(100 * (observed[:, 2].mean() - 1.255)),
                          measured_minus_prediction_z_cm=float(100 * (observed[:, 2].mean() - prediction_z)),
                          mocap_minus_logger_z_mean_mm=float(1000 * delta_stream[:, 2].mean()),
                          mocap_minus_logger_z_rms_mm=float(1000 * np.sqrt(np.mean(delta_stream[:, 2] ** 2))))
            row['phases'][name] = values
            axes[j].scatter(i + 1, observed[:, 2].mean(), color='#ea580c', s=65,
                            label='Measured OptiTrack origin' if i == 0 else None)
            axes[j].scatter(i + 1, prediction_z, marker='x', color='#2563eb', s=65,
                            label='Saved nominal start' if j == 0 and i == 0 else ('Saved prediction' if i == 0 else None))
            axes[j].annotate(f'+{values["measured_minus_prediction_z_cm"]:.1f} cm',
                             (i + 1, observed[:, 2].mean()), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=9)
        rows.append(row)
    for ax, title in zip(axes, ['Before whip: last 0.21 s', 'Final hover: 9.2–10.9 s into CSV']):
        ax.axhline(1.255, color='#1f2937', linestyle='--', label='Command: 1.255 m')
        ax.set(title=title, xlabel='Flight', ylabel='Tracked-origin height [m]', xticks=range(1, 6), xlim=(.6, 5.4), ylim=(1.24, 1.35))
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, .25), fontsize=8)
    figure.suptitle('adp0: measured hover is higher than commanded / predicted', fontsize=13)
    figure.savefig(OUTPUT / 'hover-height.png', dpi=180)
    figure.savefig(OUTPUT / 'hover-height.pdf')
    plt.close(figure)
    changed = [p for row in rows for p, digest in row['hashes'].items() if sha256(p) != digest]
    assert not changed
    result = dict(rows=rows, input_files_unchanged=True, no_fitting_training_or_spatial_shift=True,
                  origin_convention='Both measured and predicted drone positions are cf_7 tracked origins. The 55 mm cable-attachment offset is separate.',
                  logger_limitation='Logger stores cf.get_position(); no independently identified onboard estimator state or thrust/integrator is recorded.',
                  fitting_initialization='M1 fitting uses measured causal prehover state and inferred frozen effective compensation.',
                  deployment_initialization='Saved rehearsal and current training use ideal settled initial state with zero effective compensation.')
    (OUTPUT / 'hover_height.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for row in rows:
        print(row['take'], {name: round(values['measured_minus_prediction_z_cm'], 3) for name, values in row['phases'].items()})


if __name__ == '__main__':
    main()
