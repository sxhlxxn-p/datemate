import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PipelineTests(unittest.TestCase):
    def test_fictional_end_to_end_and_pair_holdout(self):
        builder = load('build_common_dataset')
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source'
            load('make_synthetic_data').generate(source)
            audit, visits = builder.load_sources(source)
            builder.resolve_places(visits, {})
            places = builder.build_places(visits, {})
            rows, _ = builder.prepare_interactions(visits, places)
            users = builder.build_users(audit['_users'], rows)
            output = Path(tmp) / 'processed'
            output.mkdir()
            for name, table in [('interactions', rows), ('places', places), ('users', users)]:
                builder.write_csv(output / (name + '.csv'), table)
            self.assertEqual(load('validate_common_dataset').validate(output)['interactions'], 20)
            # A repeated visit to a held-out place must stay in that place's split.
            repeat = dict(visits[0], interaction_id='SYNTH_REPEAT', VISIT_START_YMD='2022-12-31')
            repeated, _ = builder.prepare_interactions(visits + [repeat], places)
            same_pair = [r for r in repeated if r['user_id'] == repeat['user_id']
                         and r['place_id'] == repeat['place_id']]
            self.assertEqual(len({r['known_user_split'] for r in same_pair}), 1)
            # The validator must reject a real pair split leak.
            damaged = dict(rows[0], interaction_id='SYNTH_BAD', known_user_split='test')
            if rows[0]['known_user_split'] == 'test':
                damaged['known_user_split'] = 'train'
            builder.write_csv(output / 'interactions.csv', rows + [damaged])
            with self.assertRaisesRegex(AssertionError, 'User-place leakage'):
                load('validate_common_dataset').validate(output)


if __name__ == '__main__':
    unittest.main()
