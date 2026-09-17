"""Practical stopping: best weights and meaningful-progress patience are distinct."""
from dataclasses import dataclass
import math


@dataclass
class Plateau:
    minimum: int
    patience: int = 5
    relative: float = .005
    best: float = math.inf
    anchor: float = math.inf
    stale: int = 0

    def observe(self, update, score):
        if not math.isfinite(score):
            raise FloatingPointError('Nonfinite model selection loss')
        improved=score<self.best
        self.best=min(self.best,score)
        if not math.isfinite(self.anchor) or score < self.anchor-max(abs(self.anchor)*self.relative,1e-9):
            self.anchor=score;self.stale=0
        else:
            self.stale+=1
        return improved, update>=self.minimum and self.stale>=self.patience
