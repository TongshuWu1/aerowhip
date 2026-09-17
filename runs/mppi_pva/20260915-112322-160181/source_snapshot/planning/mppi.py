"""Offline information-theoretic path-integral search in spline coordinates.

Fixed Gaussian prior at the seed, fixed covariance, shifted Gaussian proposals.
Only independent proposal samples enter the importance-weighted update. The
incumbent and current mean are evaluated separately, never treated as samples.
This is a spline-parameterized offline variant, not aircraft feedback MPC.
"""
import numpy as np


def importance_weights(samples, scores, mean, prior, std, temperature):
    """p*(z) ∝ exp(score(z)/temperature) N(z; prior, diag(std**2))."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('MPPI temperature must be finite and positive.')
    samples, scores, mean, prior, std = map(np.asarray, (samples, scores, mean, prior, std))
    valid = np.isfinite(scores)
    weights = np.zeros(len(scores))
    if not valid.any():
        return weights
    # log(p/q), dropping constants common to all samples; using the difference
    # form avoids subtracting large squared world coordinates.
    delta = (mean-prior)/std
    log_ratio = -((samples[valid]-mean)/std)@delta - .5*np.dot(delta, delta)
    log_w = (scores[valid]-np.max(scores[valid]))/temperature + log_ratio
    log_w -= np.max(log_w)
    weights[valid] = np.exp(log_w)
    weights /= weights.sum()
    return weights


def optimize(mean, std, evaluate, *, population=64, iterations=12, temperature=20.,
             seed=655, progress=None, cancelled=None):
    mean, std = np.array(mean, float), np.array(std, float)
    if (mean.ndim != 1 or mean.shape != std.shape or not len(mean)
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()
            or population < 4 or iterations < 1 or not np.isfinite(temperature) or temperature <= 0):
        raise ValueError('Invalid MPPI mean, covariance, population, iterations or temperature.')
    prior = mean.copy()
    rng = np.random.default_rng(seed)
    best, best_score, history, bank = mean.copy(), -np.inf, [], []
    for iteration in range(iterations):
        if cancelled and cancelled():
            break
        samples = mean + rng.standard_normal((population, len(mean)))*std
        # One GPU batch; deterministic rows excluded from importance weights.
        candidates = np.concatenate([samples, mean[None], best[None]])
        scores, diagnostics = evaluate(candidates)
        scores = np.asarray(scores, float)
        if scores.shape != (len(candidates),):
            raise ValueError('Evaluator returned the wrong number of scores.')
        finite = np.flatnonzero(np.isfinite(scores))
        if not len(finite):
            raise ValueError('No feasible MPPI candidates, including the seed/mean. Check launch, duration and constraints; no CSV exported.')
        order = finite[np.argsort(scores[finite])[::-1]]
        if scores[order[0]] > best_score:
            best, best_score = candidates[order[0]].copy(), float(scores[order[0]])
        bank.extend((float(scores[i]), candidates[i].copy()) for i in order[:32])
        unique = {}
        for score, vector in sorted(bank, key=lambda entry: entry[0], reverse=True):
            unique.setdefault(vector.tobytes(), (score, vector))
        bank = list(unique.values())[:32]
        weights = importance_weights(samples, scores[:population], mean, prior, std, temperature)
        if weights.sum() > 0:
            mean += weights@(samples-mean)
        row = dict(diagnostics, iteration=iteration+1, best_score=best_score,
                   feasible_fraction=len(finite)/len(candidates),
                   effective_sample_size=float(1/np.dot(weights, weights)) if weights.any() else 0.,
                   maximum_weight=float(weights.max()), temperature=float(temperature),
                   proposal_samples=population, evaluated_candidates=len(candidates))
        history.append(row)
        if progress:
            progress(row, best, bank)
    return best, history, bank
