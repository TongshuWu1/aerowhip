"""Versioned task outcomes; missing versions preserve historical semantics."""
import torch

LEGACY='legacy_strike_v1'
TIP_CONTACT='tip_contact_v1'


def criterion(task):
    value=task.get('success_criterion',LEGACY)
    if value not in (LEGACY,TIP_CONTACT):raise ValueError('Unknown PVA success criterion: '+str(value))
    return value


def label(task):
    return 'Tip reaches target' if criterion(task)==TIP_CONTACT else 'Legacy speed / strategy / wave gates'


def contact_success(task,running,tip_entry,legacy_hit):
    # Feasibility is already included in running; end-of-command validation
    # can still revoke a hit. Other cable nodes do not veto a tip-contact task.
    return running&torch.isfinite(tip_entry) if criterion(task)==TIP_CONTACT else legacy_hit
