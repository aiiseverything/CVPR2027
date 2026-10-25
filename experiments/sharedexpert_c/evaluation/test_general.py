import unittest
from unittest.mock import patch
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import random
import time
from common import score, score_prediction
from general_metrics import ifeval_library, ifeval_seed, score_general
from run_api import infer


class GeneralTests(unittest.TestCase):
    def test_secondary_metrics_require_complete_benchmark(self):
        from general_metrics import aggregate_general
        rows = [{'id':'a','task':'ifeval'},{'id':'b','task':'ifeval'}]
        a = {'correct':True,'loose_correct':True,'strict_instructions':[True,True],
             'loose_instructions':[True,True]}
        b = {'correct':False,'loose_correct':True,'strict_instructions':[False],
             'loose_instructions':[True]}
        self.assertEqual(aggregate_general(rows,{'m':{'a':a}}),{})
        result = aggregate_general(rows,{'m':{'a':a,'b':b}})['m']['ifeval']
        self.assertEqual(result['strict_prompt_accuracy'],0.5)
        self.assertAlmostEqual(result['strict_instruction_accuracy'],2/3)
        self.assertEqual(result['loose_prompt_accuracy'],1)

    def test_ocr_alias_case_and_math_normalization(self):
        row = {'task':'ocrbench','category':'Regular Text Recognition','answers':['Cat','Kitten']}
        self.assertTrue(score(row,'</think>CAT')['correct'])
        self.assertFalse(score(row,'<think>Cat')['correct'])
        row.update(category='Handwritten Mathematical Expression Recognition',answers=['x + y'])
        self.assertTrue(score(row,'</think>x+y')['correct'])
        self.assertFalse(score(row,'</think>X+y')['correct'])

    def test_ifeval_whole_response_not_last_line(self):
        row = {'task':'ifeval','ifeval':{'key':1,'prompt':'Answer in JSON.',
               'instruction_id_list':['detectable_format:json_format'],'kwargs':[{}]}}
        self.assertTrue(score(row,'</think>{"answer": 3}')['correct'])
        self.assertFalse(score(row,'</think>Unwanted commentary\n{"answer": 3}')['correct'])
        self.assertFalse(score(row,'<think>{"answer":3}')['correct'])

    def test_length_zero_applies_to_if_diagnostics(self):
        row = {'task':'ifeval','ifeval':{'key':1,'prompt':'Answer in JSON.',
               'instruction_id_list':['detectable_format:json_format'],'kwargs':[{}]}}
        record = score_prediction(row, {'raw_response':'{"a":3}','reasoning':'done','finish_reason':'length'})
        self.assertTrue(record['correct_before_length_penalty'])
        self.assertFalse(record['correct'])
        self.assertFalse(record['loose_correct'])
        self.assertEqual(record['strict_instructions'],[False])

    def test_text_only_request_has_no_image_processor_kwargs(self):
        args = SimpleNamespace(model='test',name='baseline',retries=1,timeout=1,endpoints=['http://localhost/v1'])
        row = {'id':'x','task':'exact','answer':'yes','images':[],'image_file_sha256':[],
               'prompt':'hi','prompt_tokens':4}
        response = {'choices':[{'message':{'content':'yes','reasoning':'done'},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':4}}
        with patch('run_api.request_json',return_value=response) as request:
            self.assertTrue(infer(row,args)['correct'])
            self.assertNotIn('mm_processor_kwargs',request.call_args.args[1])


    def test_model_names_are_shared_with_remote_runners(self):
        import common
        import model_names
        self.assertIs(common.MODEL_NAMES, model_names.MODEL_NAMES)
        self.assertEqual(len(model_names.MODEL_NAMES), len(set(model_names.MODEL_NAMES)))
        self.assertTrue({'baseline-9b', 'perception-v2-step100', 'domain-opd', 'mix-rl'}
                        <= set(model_names.MODEL_NAMES))

    def test_text_only_request_retains_multi_endpoint_failover(self):
        import hashlib
        import urllib.error
        endpoints = ['http://127.0.0.1:8001/v1', 'http://127.0.0.1:8002/v1']
        args = SimpleNamespace(model='test', name='baseline-9b', retries=2,
                               timeout=1, endpoints=endpoints)
        row = {'id':'text-only-failover', 'task':'exact', 'answer':'yes',
               'images':[], 'image_file_sha256':[], 'prompt':'hi', 'prompt_tokens':4}
        response = {'choices':[{'message':{'content':'yes','reasoning':'done'},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':4}}
        with patch('run_api.request_json', side_effect=[urllib.error.URLError('offline'), response]) as request, \
             patch('run_api.time.sleep'):
            self.assertTrue(infer(row, args)['correct'])
        start = int.from_bytes(hashlib.sha256(row['id'].encode()).digest()[:8], 'big') % 2
        self.assertEqual([call.args[0] for call in request.call_args_list],
                         [endpoints[start]+'/chat/completions',
                          endpoints[(start+1)%2]+'/chat/completions'])
        self.assertTrue(all('mm_processor_kwargs' not in call.args[1]
                            for call in request.call_args_list))


class DeterministicIFEvalTests(unittest.TestCase):
    """Fixtures reproduce both unsupported-symbol branches of the pinned checker."""
    def setUp(self):
        state = random.getstate()
        self.addCleanup(random.setstate, state)
        self.rows = [
            {'task': 'ifeval', 'ifeval': {
                'key': key, 'prompt': f'Include at least {count} {symbol} characters.',
                'instruction_id_list': ['keywords:letter_frequency'],
                'kwargs': [{'letter': symbol, 'let_frequency': count, 'let_relation': 'at least'}],
            }} for key, symbol, count in [(1122, '#', 4), (1129, '!', 6)]
        ]

    @staticmethod
    def random_checker(example, responses):
        bits = random.getrandbits(128)
        time.sleep(0.002)  # Make concurrent state-save/restore races observable.
        flags = [bool(bits & (1 << index)) for index in range(128)]
        return SimpleNamespace(follow_all_instructions=all(flags), follow_instruction_list=flags)

    def fake_library(self):
        return SimpleNamespace(InputExample=SimpleNamespace,
            test_instruction_following_strict=self.random_checker,
            test_instruction_following_loose=self.random_checker)

    def test_same_identity_ignores_model_response_and_dictionary_order(self):
        row = self.rows[0]
        reordered = {**row, 'id': 'presentation-only-id', 'model': 'another-model',
                     'ifeval': dict(reversed(list(row['ifeval'].items())))}
        self.assertEqual(ifeval_seed(row), ifeval_seed(reordered))
        self.assertNotEqual(ifeval_seed(row), ifeval_seed(self.rows[1]))
        with patch('general_metrics.ifeval_library', return_value=self.fake_library()):
            a = score_general(row, 'first response')
            b = score_general(reordered, 'different response')
        self.assertEqual(a['ifeval_random_seed'], b['ifeval_random_seed'])
        self.assertEqual(a['strict_instructions'], b['strict_instructions'])

    def test_repeated_reordered_and_concurrent_calls_restore_rng(self):
        with patch('general_metrics.ifeval_library', return_value=self.fake_library()):
            expected = [score_general(row, 'answer') for row in self.rows]
            for result in expected:
                self.assertEqual(result['strict_instructions'], result['loose_instructions'])
            for caller_seed in [0, 711, 20261019]:
                random.seed(caller_seed)
                before = random.getstate()
                order = [1, 0, 1, 0]
                self.assertEqual([score_general(self.rows[i], 'answer') for i in order],
                                 [expected[i] for i in order])
                self.assertEqual(random.getstate(), before)
            before = random.getstate()
            order = [1, 0] * 12
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda i: score_general(self.rows[i], 'answer'), order))
            self.assertEqual(results, [expected[i] for i in order])
            self.assertEqual(random.getstate(), before)

    def test_actual_symbol_fallback_repeats_and_matches_strict_loose(self):
        ifeval_library()  # Load and verify the unchanged pinned upstream files.
        from instruction_following_eval.instructions import LetterFrequencyChecker
        original = LetterFrequencyChecker.build_description
        selected = []

        def record_selection(instance, **kwargs):
            description = original(instance, **kwargs)
            selected.append(instance.get_instruction_args()['letter'])
            return description

        with patch.object(LetterFrequencyChecker, 'build_description', record_selection):
            for row in self.rows:
                selections = []
                for caller_seed in [1, 123, 9876]:
                    random.seed(caller_seed)
                    before = random.getstate()
                    selected.clear()
                    result = score_general(row, 'aaaaaa #### !!!!!!')
                    self.assertEqual(len(selected), 2)
                    self.assertEqual(selected[0], selected[1])
                    self.assertTrue(selected[0].isascii() and selected[0].isalpha())
                    self.assertEqual(result['strict_instructions'], result['loose_instructions'])
                    self.assertEqual(random.getstate(), before)
                    selections.append(selected[0])
                self.assertEqual(len(set(selections)), 1)

    def test_strict_and_loose_exceptions_restore_rng(self):
        def fail_after_draw(*args):
            random.getrandbits(128)
            raise RuntimeError('checker failure')

        for name in ['test_instruction_following_strict', 'test_instruction_following_loose']:
            with self.subTest(checker=name):
                library = self.fake_library()
                setattr(library, name, fail_after_draw)
                before = random.getstate()
                with patch('general_metrics.ifeval_library', return_value=library):
                    with self.assertRaisesRegex(RuntimeError, 'checker failure'):
                        score_general(self.rows[0], 'answer')
                self.assertEqual(random.getstate(), before)


if __name__ == '__main__':
    unittest.main()
