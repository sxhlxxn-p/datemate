"""Reproducible synthetic-only strategy-selection experiment.

Frozen holdout: no feedback updates on test. Each seed uses identical contexts,
latent candidate utilities and potential outcomes for every policy. Simulator
oracle is diagnostic only, never passed to policy selection or updates.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datemate.bandit import ACTIONS, ContextualBandit


def environment(seed, rounds, candidates, slate_size, scenario, alpha):
    rng = np.random.default_rng(seed)
    gap = rng.integers(0, 2, rounds)
    history = rng.integers(0, 2, rounds)
    attributes = rng.integers(0, 2, rounds)
    contexts = gap * 4 + history * 2 + attributes
    a = rng.uniform(.05, .95, (rounds, candidates))
    independent = rng.uniform(.05, .95, (rounds, candidates))
    b = np.where(gap[:, None] == 1, independent, np.clip(a + rng.normal(0, .08, a.shape), .01, .99))
    latent = np.stack([a, b], axis=1)
    # Synthetic model estimates are noisy measurements; no fitted real model.
    cbf_noise = np.where(attributes, .12, .5)[:, None, None]
    cf_noise = np.where(history, .08, .6)[:, None, None]
    if scenario == 'noisy':
        cbf_noise = cbf_noise * 2
        cf_noise = cf_noise * 2
    cbf = np.clip(latent + rng.normal(size=latent.shape) * cbf_noise, 0, 1)
    cf = np.clip(latent + rng.normal(size=latent.shape) * cf_noise, 0, 1)
    supported = rng.random(latent.shape) < np.where(history, .9, .1)[:, None, None]
    if scenario == 'uninformative':
        cbf = rng.random(latent.shape)
        cf = rng.random(latent.shape)
    hybrid = np.where(supported, alpha * cf + (1-alpha) * cbf, cbf)
    scores = np.stack([cbf.mean(1), cbf.min(1), hybrid.mean(1), hybrid.min(1)], axis=1)
    # Stable tie order is candidate index; existing pair scorer has equivalent ordering.
    slates = np.argsort(-scores, axis=2, kind='stable')[:, :, :slate_size]
    joint = a * b
    expected = np.take_along_axis(joint[:, None, :], slates, axis=2).mean(2)
    observed = (rng.random(expected.shape) < expected).astype(float)
    return contexts, expected, observed


def run_seed(config, scenario, seed, event_sink=None):
    train, test = config['train_rounds'], config['test_rounds']
    contexts, expected, observed = environment(seed, train+test, config['candidates'],
        config['slate_size'], scenario, config['hybrid_alpha'])
    policies = {name: ContextualBandit(8, config['epsilon'], seed+10000+i)
                for i, name in enumerate(('contextual', 'noncontextual'))}
    # Matched action-level potential outcomes across policies; only selected feedback used.
    training = {}
    for name, policy in policies.items():
        total, regret = 0., 0.
        for t in range(train):
            ctx = int(contexts[t]) if name == 'contextual' else 0
            action, probability = policy.select(ctx)
            reward = float(observed[t, action])
            if event_sink:
                event_sink([scenario, seed, "train", name, t, ctx, ACTIONS[action], probability, reward])
            policy.update(ctx, action, reward)
            total += reward
            regret += expected[t].max() - expected[t, action]
        training[name] = {'reward_mean': total/train, 'expected_regret_mean': regret/train}
    metrics = {}
    rng = np.random.default_rng(seed+20000)
    for name in [*ACTIONS, 'random', *policies, 'oracle']:
        actions, probabilities = [], []
        for t in range(train, train+test):
            if name in ACTIONS:
                action, probability = ACTIONS.index(name), 1.
            elif name == 'random':
                action, probability = int(rng.integers(4)), .25
            elif name == 'oracle':
                action, probability = int(expected[t].argmax()), 1.
            else:
                ctx = int(contexts[t]) if name == 'contextual' else 0
                action, probability = policies[name].select(ctx, explore=False)
            if event_sink and name != "oracle":
                event_sink([scenario, seed, "test", name, t-train, int(contexts[t]), ACTIONS[action], probability, float(observed[t, action])])
            actions.append(action); probabilities.append(probability)
        actions = np.asarray(actions); ts = np.arange(train, train+test)
        chosen_expected = expected[ts, actions]
        metrics[name] = {'reward_mean': float(observed[ts, actions].mean()),
            'expected_reward_mean': float(chosen_expected.mean()),
            'expected_regret_mean': float((expected[ts].max(1)-chosen_expected).mean()),
            'action_counts': np.bincount(actions, minlength=4).tolist()}
    return {'seed': seed, 'training': training, 'test': metrics}


def experiment(config, event_sink=None):
    if config['seeds'] < 2 or min(config['train_rounds'], config['test_rounds']) < 1:
        raise ValueError('Need >=2 seeds and positive train/test rounds')
    if not 1 <= config['slate_size'] <= config['candidates'] or not 0 <= config['hybrid_alpha'] <= 1:
        raise ValueError('Invalid slate size or alpha')
    if not set(config['scenarios']) <= {'informative', 'noisy', 'uninformative'}:
        raise ValueError('Unknown scenario')
    output = {'config': config, 'python': platform.python_version(), 'numpy': np.__version__,
              'scope': 'synthetic_only_no_real_user_effect_estimate', 'scenarios': {}}
    for scenario in config['scenarios']:
        runs = [run_seed(config, scenario, config['seed_start']+i, event_sink) for i in range(config['seeds'])]
        aggregate = {}
        for name in runs[0]['test']:
            aggregate[name] = {}
            for key in ('reward_mean','expected_reward_mean','expected_regret_mean'):
                values = np.array([r['test'][name][key] for r in runs])
                aggregate[name][key] = {'mean': float(values.mean()), 'sd_across_seeds': float(values.std(ddof=1))}
        # Paired seed differences; bootstrap interval concerns simulator seeds only.
        differences = {}
        for baseline in [*ACTIONS, 'random', 'noncontextual']:
            values = np.array([r['test']['contextual']['expected_reward_mean']-r['test'][baseline]['expected_reward_mean'] for r in runs])
            rng = np.random.default_rng(42000)
            boot = rng.choice(values, size=(5000, len(values)), replace=True).mean(1)
            differences[baseline] = {'mean': float(values.mean()), 'seed_bootstrap_95_interval': np.quantile(boot,[.025,.975]).tolist()}
        output['scenarios'][scenario] = {'aggregate': aggregate, 'paired_expected_reward_difference': differences, 'runs': runs}
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/bandit_simulation.json')
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/bandit_simulation')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir/"synthetic_events.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario", "seed", "phase", "policy", "round", "context", "action", "propensity", "reward"])
        result = experiment(config, writer.writerow)
    result['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    result['source_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [ROOT/'datemate/bandit.py', Path(__file__).resolve()]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    for scenario, rows in result['scenarios'].items():
        print(scenario, {p: round(v['expected_reward_mean']['mean'], 4) for p,v in rows['aggregate'].items()}, flush=True)


if __name__ == '__main__':
    main()
