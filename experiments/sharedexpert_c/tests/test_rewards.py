import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('rewards', Path(__file__).parents[1]/'rewards.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


class RewardTests(unittest.TestCase):
    def score(self, text, task='exact', answers=None, **extra):
        gold={'task':task,'answers':answers or ['2'],'format':'boxed'}
        return r.compute_score('test',text,json.dumps(gold),extra)['score']

    def test_no_reasoning_credit(self):
        self.assertEqual(self.score(r'<think>\boxed{2}'),0)
        self.assertEqual(self.score(r'<think>\boxed{2}</think>\boxed{3}'),0)
        self.assertEqual(self.score(r'<think>3</think>\boxed{2}'),1)

    def test_complete_answer_and_truncation(self):
        self.assertEqual(self.score(r'answer 2'),0)
        self.assertEqual(self.score(r'\boxed{2',),0)
        self.assertEqual(self.score(r'\boxed{2} revised: \boxed{3'),0)
        self.assertEqual(self.score(r'\boxed{2}',truncated=True),0)

    def test_nested_box(self):
        self.assertEqual(r.last_boxed(r'\boxed{\frac{1}{2}}'),r'\frac{1}{2}')

    def test_exact_preserves_sign_operators_and_letters(self):
        self.assertEqual(self.score(r'\boxed{-2}'),0)
        self.assertEqual(self.score(r'\boxed{1+2}',answers=['1-2']),0)
        self.assertEqual(self.score(r'\boxed{the}',answers=['A']),0)
        self.assertEqual(self.score(r'\boxed{a}',answers=['A']),1)
        self.assertEqual(self.score(r'\boxed{two}'),1)
        self.assertEqual(self.score(r'\boxed{1,2}',answers=['12']),0)

    def test_math_rejects_partial_expression(self):
        self.assertFalse(r.math_equal(r'2\\3','2'))
        self.assertFalse(r.math_equal('$2$ $3$','2'))
        self.assertFalse(r.math_equal('', ''))

    def test_complete_single_variable_root_sets(self):
        gold = '$x_1=-3$, $x_2=2$'
        for pred in ('x_1=-3, x_2=2', '-3,2', '2,-3', 'x=-3, x=2',
                     r'x_{1}=2, x_{2}=-3', r'\{-3,2\}'):
            self.assertTrue(r.math_equal(pred, gold), pred)
        for pred in ('-3', '2', '-3,2,99', '-3,-3', '3,2', 'y_1=-3,y_2=2',
                     'x=-3,y=2', r'-3,2\\99', '$-3$ $2$', '-3,2,'):
            self.assertFalse(r.math_equal(pred, gold), pred)
        self.assertTrue(r.math_equal(r'\frac{1}{2},-2', r'$x_1=-2$, $x_2=0.5$'))
        self.assertFalse(r.math_equal('2,1', '$x=1$, $y=2$'))
        self.assertFalse(r.math_equal('2,1', '(1,2)'))

    def test_plain_math_lists_keep_order_and_multiplicity(self):
        self.assertTrue(r.math_equal('5, 8', '5,8'))
        self.assertFalse(r.math_equal('8,5', '5,8'))
        self.assertFalse(r.math_equal('-3,-3,-4', '-3,-4,-3'))
        self.assertFalse(r.math_equal('-3,-4', '-3,-4,-3'))
        self.assertTrue(r.math_equal(r'0.5,8', r'\frac{1}{2},8'))
        self.assertTrue(r.math_equal('4000', '4,000'))
        self.assertFalse(r.math_equal('4,0', '4,000'))
        self.assertFalse(r.math_equal('x=2,y=1', '$x=1$, $y=2$'))

    def test_trusted_boxed_math_references_and_degrees(self):
        self.assertTrue(r.math_equal('6', r'\boxed{6}'))
        self.assertTrue(r.math_equal('(2,4)', r'\boxed{(2,4)}'))
        self.assertTrue(r.math_equal('21', r'\boxed{$21 \mathrm{~m}$}'))
        self.assertTrue(r.math_equal('60', r'\boxed{60°}'))
        self.assertTrue(r.math_equal(r'60^\circ', '60°'))
        self.assertFalse(r.math_equal(r'6\\7', r'\boxed{6}'))

    def test_vqa_consensus_multiplicity(self):
        self.assertAlmostEqual(r.vqa_consensus('two',['2']*3+['3']*7),.9)
        self.assertEqual(r.vqa_consensus('2',['2']*4+['3']*6),1)
        self.assertEqual(r.vqa_consensus('7',['2']*10),0)

    def test_json_requires_content_and_exact_schema(self):
        gold=json.dumps({'task':'json','answers':['two'],'format':'json','json_key':'answer'})
        f=lambda s:r.compute_score('test',s,gold)['score']
        self.assertEqual(f('{"answer":"2"}'),1)
        self.assertEqual(f('{"answer":"3"}'),0)
        self.assertEqual(f('{"answer":"2","extra":1}'),0)
        self.assertEqual(f('```json\n{"answer":"2"}\n```'),0)
        self.assertEqual(f('{"answer":2}'),0)
        self.assertEqual(f('{"answer":"3","answer":"2"}'),0)
        self.assertEqual(f('{"answer":NaN}'),0)

    def test_bbox_bounds_and_overlap(self):
        self.assertEqual(r.bbox_iou('[0,0,100,100]',[0,0,100,100]),1)
        self.assertEqual(r.bbox_iou('[100,0,0,100]',[0,0,100,100]),0)
        self.assertEqual(r.bbox_iou('[-1,0,100,100]',[0,0,100,100]),0)
        self.assertAlmostEqual(r.bbox_iou('[0,0,100,100]',[50,0,150,100]),1/3)

    def test_json_retains_vqa_consensus(self):
        gold=json.dumps({'task':'json','answers':['two']+['three']*9,
                         'format':'json','json_key':'answer','content_task':'vqa'})
        self.assertAlmostEqual(r.compute_score('test','{"answer":"two"}',gold)['score'],.3)

    def test_numeric_and_document_metrics(self):
        self.assertTrue(r.math_equal('0.5','1/2'))
        self.assertTrue(r.math_equal('16','$$16$$'))
        self.assertTrue(r.math_equal(r'\frac{1}{2}',r'\(0.5\)'))
        self.assertEqual(r.relaxed_numeric('104',['100']),1)
        self.assertEqual(r.relaxed_numeric('106',['100']),0)
        self.assertEqual(r.relaxed_numeric('50%',['0.5']),1)
        self.assertAlmostEqual(r.anls('hello',['hallo']),.8)
        self.assertEqual(r.anls('xx',['ab']),0)


if __name__ == '__main__': unittest.main()
