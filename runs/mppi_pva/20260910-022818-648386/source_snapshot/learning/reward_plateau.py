"""Optional stopping on sustained validation-reward stability, not training noise."""
from collections import deque
import math
from statistics import mean


class RewardPlateau:
    @classmethod
    def from_history(cls, settings, *, start_episodes, history=None):
        """Continue stopping statistics without counting a repeated validation.

        History is embedded in the new run's immutable configuration, never
        read from a mutable parent worker while training.
        """
        if history is None:
            return cls(settings, start_episodes=start_episodes)
        if (history.get('schema') != 'reward_plateau_resume_v1'
                or int(history['checkpoint_episodes']) != int(start_episodes)
                or not 0 <= int(history['original_start_episodes']) <= int(start_episodes)):
            raise ValueError('Reward stopping history does not match the resumed checkpoint.')
        monitor = cls(settings, start_episodes=int(history['original_start_episodes']))
        for record in history['records']:
            if int(record['training_episodes']) > start_episodes:
                raise ValueError('Reward stopping history extends beyond the resumed checkpoint.')
            monitor.observe(record)
        monitor.decision.update(resumed_at_episodes=int(start_episodes),
            inherited_evaluations=len(monitor.seen))
        return monitor

    def __init__(self, settings, *, start_episodes):
        self.settings = dict(settings or {})
        self.enabled = bool(self.settings.get('enabled', False))
        self.start = int(start_episodes)
        self.window = int(self.settings.get('window_evaluations', 20))
        self.patience = int(self.settings.get('patience_episodes', 102400))
        self.minimum = int(self.settings.get('minimum_additional_episodes', 102400))
        self.absolute = float(self.settings.get('minimum_reward_improvement', 2.0))
        self.relative = float(self.settings.get('minimum_relative_improvement', .01))
        if self.window < 4 or self.window % 2 or self.patience <= 0 or self.minimum < 0:
            raise ValueError('Reward stopping needs an even window >= 4 and positive patience.')
        if not all(math.isfinite(x) and x >= 0 for x in (self.absolute, self.relative)):
            raise ValueError('Reward stopping tolerances must be finite and nonnegative.')
        self.values = deque(maxlen=self.window)
        self.seen = set()
        self.anchor = None
        self.best_mean = None
        self.last_improvement = self.start
        self.last_step = self.start - 1
        self.decision = dict(enabled=self.enabled, status='WARMUP' if self.enabled else 'DISABLED',
                             should_stop=False, start_episodes=self.start, settings=self.settings)

    def observe(self, record):
        if not self.enabled:
            return dict(self.decision)
        step = int(record['training_episodes'])
        identity = record.get('evaluation_id', step)
        reward = record.get('mean_episode_reward')
        if (step <= self.last_step or identity in self.seen or record.get('evaluation_reused')
                or reward is None or not math.isfinite(reward)):
            return dict(self.decision)
        self.seen.add(identity)
        self.last_step = step
        self.values.append(float(reward))
        self.decision.update(latest_episodes=step, additional_episodes=step-self.start,
                             window_count=len(self.values), latest_reward=float(reward))
        if len(self.values) < self.window:
            return dict(self.decision)
        current = mean(self.values)
        threshold = max(self.absolute, self.relative * abs(self.anchor or 0.))
        if self.anchor is None or current > self.anchor + threshold:
            self.anchor = current
            self.last_improvement = step
        self.best_mean = current if self.best_mean is None else max(self.best_mean, current)
        threshold = max(self.absolute, self.relative * abs(self.anchor))
        half = self.window // 2
        values = list(self.values)
        drift = abs(mean(values[half:]) - mean(values[:half]))
        stable = drift <= threshold
        near_best = current >= self.best_mean - threshold
        stale = step - self.last_improvement
        stop = (step - self.start >= self.minimum and stale >= self.patience
                and stable and near_best)
        self.decision.update(status='PLATEAU' if stop else 'MONITORING', should_stop=stop,
                             rolling_mean_reward=current, best_rolling_mean_reward=self.best_mean,
                             significant_improvement_anchor=self.anchor, threshold_reward=threshold,
                             last_improvement_episodes=self.last_improvement,
                             attempts_without_improvement=stale, window_half_drift=drift,
                             stable=stable, near_best=near_best)
        return dict(self.decision)
