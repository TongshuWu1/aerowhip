from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = Path(__file__).resolve().parents[2]


def _trajectory():
    positions = np.full((4, 4, 3), 5.0)
    positions[:, -1] = [[-.10, 0, 0], [.10, 0, 0], [.20, 0, 0], [.30, 0, 0]]
    velocities = np.zeros_like(positions)
    velocities[:, -1, 0] = 5.0
    return positions, velocities


def _diagnose(positions, velocities):
    from simulator.strike_diagnostics import replay_strike_diagnostics
    return replay_strike_diagnostics(positions, velocities, [2, 3], np.zeros(3), np.array([1, 0, 0]),
        radius_m=.05, minimum_speed_m_s=4, maximum_angle_deg=45)


def test_fast_crossing_is_a_hit_even_when_both_displayed_positions_are_outside():
    positions, velocities = _trajectory()
    # Node 0/1 are attachment/internal nodes, not the cable markers checked by training.
    positions[:, :2] = 0
    result = _diagnose(positions, velocities)
    assert result['hit'].tolist() == [False, True, False, False]
    assert result['success_so_far'].tolist() == [False, True, True, True]


def test_invalid_tip_then_other_marker_disqualifies_a_later_valid_entry():
    positions, velocities = _trajectory()
    positions[:, -1] = [[-.1, 0, 0], [.1, 0, 0], [.2, 0, 0], [-.1, 0, 0]]
    velocities[1, -1] = [2, 0, 0]
    positions[2:, 2] = 0
    result = _diagnose(positions, velocities)
    assert result['tip_contact'][1]
    assert not result['hit'].any()
    assert result['disqualified'][2:].all()


def test_later_valid_tip_entry_is_allowed_after_a_slow_tip_only_entry():
    positions, velocities = _trajectory()
    positions[3, -1] = [-.1, 0, 0]
    velocities[1, -1] = [2, 0, 0]
    result = _diagnose(positions, velocities)
    assert result['hit'].tolist() == [False, False, False, True]
