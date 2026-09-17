"""PPO credit assignment for the exact saved whole-trajectory MPPI objective.

Feasible terminal scores call the shared scorer. A finite penalty represents
MPPI's rejected trajectories. Potential differences redistribute reward without
changing the undiscounted ordering; terminal potential is always zero.
"""
from pathlib import Path
import shutil
import numpy as np
import torch
from experimental_data.io import sha256_file

SCHEMA = 'mppi_preferred_fold_v1'


def enabled(cfg):
    return cfg.get('ppo_objective', {}).get('schema') == SCHEMA


def validate(cfg):
    spec = cfg.get('ppo_objective')
    if spec is None:
        return
    if not enabled(cfg) or cfg['method'] != 'ppo':
        raise ValueError('Unknown PPO trajectory objective')
    weights = cfg.get('trajectory_objective', {})
    if weights.get('schema') != 'preferred_fold_v1':
        raise ValueError('PPO requires the shared preferred-fold objective')
    if any(v < 0 for v in weights.values() if isinstance(v, (float, int))):
        raise ValueError('Trajectory weights must be nonnegative')
    for key in ('shape_sigma', 'tangent_sigma', 'proximity_scale_m'):
        if weights.get(key, 0) <= 0:
            raise ValueError('Positive trajectory scales required')
    if spec.get('failure_penalty', 0) <= 0 or spec.get('potential_scale', -1) < 0:
        raise ValueError('Positive failure penalty and nonnegative potential scale required')
    name = weights.get('reference_file', '')
    if not name or Path(name).name != name:
        raise ValueError('Objective reference must be a local file name')
    if len(spec.get('reference_sha256', '')) != 64:
        raise ValueError('Frozen objective reference checksum required')
    selection = spec.get('selection_criterion', 'reward_v1')
    if selection == 'reward_v1':
        if cfg['training'].get('success_priority', False):
            raise ValueError('Historical matched MPPI objective ranks policies by reward')
    elif selection == 'tip_contact_then_score_v1':
        if not cfg['training'].get('success_priority', False):
            raise ValueError('Matched contact-first selection requires success priority')
        if cfg['task'].get('success_criterion') != 'tip_contact_v1':
            raise ValueError('Matched contact-first selection requires tip contact')
        if any(cfg['launch'].get(key, 0) != 0 for key in ('start_radius_m', 'target_radius_m')):
            raise ValueError('Matched contact-first selection requires the fixed MPPI scenario')
    else:
        raise ValueError('Unknown matched PPO checkpoint selection')


def freeze_reference(cfg, directory, *, source_root=None):
    if not enabled(cfg):
        return
    name = cfg['trajectory_objective']['reference_file']
    source = Path(source_root)/name if source_root is not None else Path(cfg['ppo_objective']['reference_source'])
    if sha256_file(source) != cfg['ppo_objective']['reference_sha256']:
        raise ValueError('Objective reference checksum differs')
    shutil.copy2(source, Path(directory)/name)


def redistribute(score, potentials, masks, dones):
    """[T,B,1] rewards sum to score - initial potential, including early exits."""
    if bool(((masks > 0).sum(0) == 0).any()):
        raise ValueError('Every trajectory needs an observed transition')
    terminal = (dones > 0) & (masks > 0)
    if not bool((terminal.sum(0) == 1).all()):
        raise ValueError('Whole-trajectory rewards require exactly one terminal transition')
    after = torch.where(dones.bool(), torch.zeros_like(potentials[1:]), potentials[1:])
    rewards = (after - potentials[:-1]) * masks
    return rewards + terminal * score[None, :, None]


class TrajectoryReward:
    def __init__(self, env):
        from planning.whip_objective import PreferredFoldCapture
        cfg = env.settings
        validate(cfg)
        source = Path(env.root)/cfg['trajectory_objective']['reference_file']
        if sha256_file(source) != cfg['ppo_objective']['reference_sha256']:
            raise ValueError('Frozen objective reference checksum differs')
        with np.load(source, allow_pickle=False) as data:
            reference = env.tensor(data['shape'])
        if reference.ndim != 3 or reference.shape[-1] != 3 or not bool(torch.isfinite(reference).all()):
            raise ValueError('Invalid preferred-shape reference')
        self.capture = PreferredFoldCapture(reference)
        self.potentials = []
        self.observe(env)

    def observe(self, env):
        self.capture(env)
        phi = -env.settings['ppo_objective']['potential_scale'] * env.minimum_distance / env.cable_length
        self.potentials.append(phi.clone()[:, None])

    def finish(self, env, result):
        if bool(env.active.any()):
            raise ValueError('Cannot score an unfinished PPO trajectory')
        score, terms = self.capture.score(env, result, result['actions'], env.settings['trajectory_objective'])
        score = torch.where(result['failed'], score.new_full(score.shape, -env.settings['ppo_objective']['failure_penalty']), score)
        if not bool(torch.isfinite(score).all()):
            raise ValueError('Nonfinite PPO trajectory return')
        result.update(legacy_reward=result['reward'], reward=score, objective_terms=terms)
        env.total.copy_(score)
        return result

    def training_rewards(self, result, masks, dones):
        return redistribute(result['reward'], torch.stack(self.potentials), masks, dones)
