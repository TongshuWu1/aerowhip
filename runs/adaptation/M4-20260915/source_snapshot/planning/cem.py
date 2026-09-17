"""Reproducible diagonal-Gaussian CEM with incumbent retention and finite elites."""
import numpy as np


def optimize(mean, std, evaluate, *, population=64, iterations=10, elite_fraction=.125,
             seed=655, smoothing=.25, std_floor=None, progress=None, cancelled=None):
    if population < 4 or iterations < 1 or not 0 < elite_fraction <= .5:
        raise ValueError('Invalid CEM population, iterations or elite fraction.')
    mean, std = np.array(mean, float), np.array(std, float)
    floor = np.full_like(std, .002) if std_floor is None else np.asarray(std_floor)
    rng = np.random.default_rng(seed)
    best, best_score, history, bank = mean.copy(), -np.inf, [], []
    for iteration in range(iterations):
        if cancelled and cancelled():
            break
        samples = mean + rng.standard_normal((population, len(mean)))*std
        samples[0] = best
        samples[1] = mean
        scores, diagnostics = evaluate(samples)
        scores = np.asarray(scores, float)
        finite = np.flatnonzero(np.isfinite(scores))
        if not len(finite):
            raise ValueError('No feasible spline candidates. Increase duration or reduce exploration; no CSV exported.')
        order = finite[np.argsort(scores[finite])[::-1]]
        elites = order[:max(2, int(population*elite_fraction))]
        if scores[order[0]] > best_score:
            best, best_score = samples[order[0]].copy(), float(scores[order[0]])
        for index in elites:
            bank.append((float(scores[index]), samples[index].copy()))
        unique={}
        for score,vector in sorted(bank,key=lambda entry:entry[0],reverse=True):
            unique.setdefault(vector.tobytes(),(score,vector))
        bank=list(unique.values())[:32]
        mean = smoothing*mean+(1-smoothing)*samples[elites].mean(axis=0)
        std = np.maximum(floor, smoothing*std+(1-smoothing)*samples[elites].std(axis=0))
        row = dict(iteration=iteration+1, best_score=best_score,
                   feasible_fraction=len(finite)/population, **diagnostics)
        history.append(row)
        if progress:
            progress(row, best, bank)
    return best, history, bank
