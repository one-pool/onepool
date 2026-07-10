"""DiLoCo outer optimizer: Nesterov SGD over averaged pseudo-gradients.

The coordinator owns the canonical adapter weights. Each round, workers send
pseudo-gradients (start − local); the coordinator averages them weighted by
samples processed and takes one outer step. Defaults (lr 0.7, momentum 0.9)
are the values the DiLoCo paper found robust across scales.
"""

from __future__ import annotations

import numpy as np


class NesterovOuter:
    def __init__(self, weights: dict[str, np.ndarray], lr: float = 0.7, momentum: float = 0.9):
        self.weights = {k: v.astype(np.float32, copy=True) for k, v in weights.items()}
        self.lr = lr
        self.momentum = momentum
        self._velocity = {k: np.zeros_like(v) for k, v in self.weights.items()}

    def step(self, pseudo_grad: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply one outer update; returns the new canonical weights."""
        for name, grad in pseudo_grad.items():
            v = self._velocity[name]
            v *= self.momentum
            v += grad
            # Nesterov look-ahead: step along grad + momentum-corrected velocity
            self.weights[name] -= self.lr * (grad + self.momentum * v)
        return self.weights
