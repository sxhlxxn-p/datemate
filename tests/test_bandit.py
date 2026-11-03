import copy
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from datemate.bandit import ContextualBandit, rank_strategies
from run_bandit_simulation import environment, run_seed


class BanditTests(unittest.TestCase):
    def test_probability_and_selected_feedback_only(self):
        b = ContextualBandit(seed=3)
        for _ in range(20):
            action, probability = b.select(2)
            self.assertAlmostEqual(probability, b.probabilities(2)[action])
            b.update(2, action, 1.)
        self.assertEqual(b.counts.sum(), 20)
        self.assertEqual(b.counts[:2].sum(), 0)
        self.assertEqual(b.counts[3:].sum(), 0)
        self.assertTrue(np.all(b.probabilities(2) >= .025))
        self.assertAlmostEqual(b.probabilities(2).sum(), 1.)
        b.update(0, 1, 1.)
        np.testing.assert_allclose(b.probabilities(0, False), [0,1,0,0])

    def test_validation(self):
        for epsilon in (0, -1, 1.1, float('nan')):
            with self.assertRaises(ValueError): ContextualBandit(epsilon=epsilon)
        b = ContextualBandit()
        for c,a,r in [(-1,0,1),(0,4,1),(0,0,float('nan')),(0,0,1.1)]:
            with self.assertRaises(ValueError): b.update(c,a,r)

    def test_real_score_adapter_shared_candidates_and_exclusion(self):
        r = rank_strategies([[1,.55,.2],[.2,.55,1]], [[.1,.6,.9],[.1,.7,.8]], ['p','q','r'], excluded={'r'}, k=2)
        self.assertEqual(r['cbf_mean'][0]['place_id'], 'p')
        self.assertEqual(r['cbf_min'][0]['place_id'], 'q')
        self.assertTrue(all('r' not in [x['place_id'] for x in rows] for rows in r.values()))
        with self.assertRaises(ValueError): rank_strategies([[1],[1]],[[1],[1]], ['p','p'])

    def test_frozen_test_no_update_and_reproducible_logs(self):
        config = dict(train_rounds=20, test_rounds=10, candidates=8, slate_size=2, epsilon=.1, hybrid_alpha=.4)
        events = []
        result = run_seed(config, 'informative', 100, events.append)
        self.assertEqual(result, run_seed(copy.deepcopy(config), 'informative', 100))
        # Exactly one selected action observation per training step and adaptive policy.
        train = [e for e in events if e[2]=='train']
        self.assertEqual(len(train),40)
        self.assertTrue(all(0 < e[7] <= 1 for e in events))
        # Test rewards are evaluated but never fed back to update().
        from unittest.mock import patch
        calls = []
        original = ContextualBandit.update
        def counted(policy,*args):
            calls.append(args); return original(policy,*args)
        with patch.object(ContextualBandit,'update',counted): run_seed(config,'noisy',101)
        self.assertEqual(len(calls), 40)

    def test_simulator_rewards_not_model_score_rewards(self):
        ctx, expected, observed = environment(1, 100, 10, 3, 'uninformative', .4)
        self.assertEqual(expected.shape, (100,4))
        self.assertTrue(np.all((expected>=0)&(expected<=1)))
        self.assertTrue(set(np.unique(observed)) <= {0.,1.})
        self.assertEqual(len(set(ctx)), 8)
        np.testing.assert_array_equal(expected, environment(1,100,10,3,'uninformative',.4)[1])


if __name__ == '__main__': unittest.main()
