import copy
import unittest
import numpy as np
from evaluation.task3_metrics import evaluate, validate_bundle

class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.b = dict(schema_version=1, synthetic=True, run_description='Synthetic tests only',
            labels=['A','B','Absent'], chunks=[dict(contract_title='test',chunk_id=i,chunk_text='synthetic') for i in range(3)],
            y_true=[[1,0,0],[1,1,0],[0,1,0]],
            models={'test':dict(predictions=[[0,1,0],[1,0,0],[0,1,0]])})
    def test_counts_and_undefined(self):
        c,s=evaluate(self.b)
        np.testing.assert_array_equal(c.TP,[1,1,0])
        np.testing.assert_array_equal(c.FP,[0,1,0])
        np.testing.assert_array_equal(c.FN,[1,1,0])
        self.assertAlmostEqual(s.micro_precision.iloc[0],2/3)
        self.assertAlmostEqual(s.micro_recall.iloc[0],.5)
        self.assertAlmostEqual(s.micro_f1.iloc[0],4/7)
        self.assertTrue(np.isnan(c.f1.iloc[2]))
        self.assertAlmostEqual(s.macro_f1_zero.iloc[0],(2/3+.5)/3)
    def test_no_predictions(self):
        self.b['models']['test']['predictions']=[[0,0,0]]*3
        c,s=evaluate(self.b)
        self.assertTrue(c.precision.isna().all())
        self.assertEqual(c.f1.iloc[0],0)
        self.assertTrue(s.no_positive_predictions.iloc[0])
    def test_invalid_inputs(self):
        for mutate in [lambda b:b['models']['test'].update(labels=['B','A','Absent']),
                       lambda b:b['models']['test'].update(row_ids=[['test',2],['test',1],['test',0]]),
                       lambda b:b['models']['test'].update(predictions=[[1]]),
                       lambda b:b['models']['test'].update(probabilities=[[float('nan')]*3]*3),
                       lambda b:b['chunks'].append(b['chunks'][0])]:
            b=copy.deepcopy(self.b); mutate(b)
            with self.assertRaises(ValueError): validate_bundle(b)
    def test_threshold(self):
        e=self.b['models']['test']
        e.update(probabilities=[[.5,.1,0],[.6,.9,0],[.1,.8,0]],threshold=.5,threshold_rule='>=')
        with self.assertRaises(ValueError): validate_bundle(self.b)
        e['predictions']=(np.asarray(e['probabilities'])>=.5).astype(int).tolist()
        validate_bundle(self.b)
    def test_sklearn_reference(self):
        from sklearn.metrics import precision_recall_fscore_support
        c,s=evaluate(self.b)
        p,r,f,_=precision_recall_fscore_support(self.b['y_true'], self.b['models']['test']['predictions'],zero_division=0)
        for key,values in [('precision',p),('recall',r),('f1',f)]:
            np.testing.assert_allclose(np.nan_to_num(c[key]),values)
            self.assertAlmostEqual(s['macro_'+key+'_zero'].iloc[0],values.mean())

if __name__=='__main__': unittest.main()
