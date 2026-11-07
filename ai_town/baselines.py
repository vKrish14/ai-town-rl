"""Explicitly named baselines, never mislabeled as trained policies."""

import numpy as np


class RandomPolicy:
    mode = "Random baseline"

    def act(self, observations, deterministic=False):
        return np.random.randint(0, 8, size=len(observations)), np.zeros(
            len(observations)
        )


class RulePolicy:
    mode = "Rule-based baseline"

    def act(self, observations, deterministic=False):
        actions = []
        for o in observations:
            if o[2] > 0.5 and o[14] > 0.01:
                a = 2
            elif o[2] > 0.5 and o[14] <= 0.01 and o[6] > 0.025:
                a = 5
            elif o[2] > 0.75 and o[6] <= 0.025:
                a = 0 if o[7] > 0.5 else 7
            elif o[1] < 0.7:
                a = 1
            elif o[7] < 0.5 and o[13] > 0.01 and o[6] < 0.6:
                a = 7
            elif o[6] < 0.45:
                a = 0 if o[7] > 0.5 else 7
            elif o[4] > 0.65:
                a = 4
            elif o[5] < 0.3:
                a = 3
            else:
                a = 0
            actions.append(a)
        return np.asarray(actions), np.zeros(len(actions))
