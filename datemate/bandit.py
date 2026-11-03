"""Contextual strategy selection; feedback must come from observed user responses.

No model score is interpreted as a reward. Contexts are finite, predeclared
segments, not user IDs. This is a one-step bandit, not sequential RL.
"""
import math
import numpy as np

ACTIONS = ('cbf_mean', 'cbf_min', 'hybrid_mean', 'hybrid_min')


class ContextualBandit:
    """Segmented epsilon-greedy with Beta(1, 1) smoothed reward means.

    Logging the selected probability is required for future off-policy work;
    these logs alone do not establish unbiased real-world evaluation.
    """
    def __init__(self, contexts=8, epsilon=.1, seed=42):
        if not isinstance(contexts, int) or contexts < 1 or not 0 < epsilon <= 1:
            raise ValueError('Use positive contexts and 0 < epsilon <= 1')
        self.epsilon = float(epsilon)
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros((contexts, len(ACTIONS)), dtype=int)
        self.rewards = np.zeros_like(self.counts, dtype=float)

    def _context(self, context):
        if not isinstance(context, (int, np.integer)) or not 0 <= context < len(self.counts):
            raise ValueError('Unknown context segment')

    def probabilities(self, context, explore=True):
        self._context(context)
        values = (self.rewards[context] + 1) / (self.counts[context] + 2)
        best = np.isclose(values, values.max(), rtol=0, atol=1e-12)
        epsilon = self.epsilon if explore else 0.
        return np.full(len(ACTIONS), epsilon / len(ACTIONS)) + (1-epsilon)*best/best.sum()

    def select(self, context, explore=True):
        probabilities = self.probabilities(context, explore)
        action = int(self.rng.choice(len(ACTIONS), p=probabilities))
        return action, float(probabilities[action])

    def update(self, context, action, reward):
        self._context(context)
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(ACTIONS):
            raise ValueError('Unknown action')
        if not math.isfinite(reward) or not 0 <= reward <= 1:
            raise ValueError('Observed reward must be finite and in [0, 1]')
        self.counts[context, action] += 1
        self.rewards[context, action] += reward


def rank_strategies(cbf_scores, hybrid_scores, candidate_ids, excluded=(), k=10):
    """Return four rankings from two aligned 2 x N normalized score arrays.

    Supply scores from frozen CBF/Hybrid models and exclude BOTH histories.
    Uses the same candidate and mean/min contract as the existing couple demo.
    No fitting, feedback, or reward inference is done here.
    """
    from datemate.hybrid import couple_recommendations
    ids = [str(p) for p in candidate_ids]
    if len(set(ids)) != len(ids):
        raise ValueError('Candidate IDs must be unique')
    rankings = {}
    for name, scores in [('cbf', cbf_scores), ('hybrid', hybrid_scores)]:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (2, len(ids)):
            raise ValueError('Scores must be aligned 2 x candidates')
        rows = [dict(user_id=u, place_id=p, score=float(scores[i, j]),
                     year='frozen', split='bandit_demo')
                for i, u in enumerate(('a', 'b')) for j, p in enumerate(ids)]
        for method in ('mean', 'min'):
            rankings[f'{name}_{method}'] = couple_recommendations(
                rows, 'a', 'b', method, k, excluded)
    return rankings
