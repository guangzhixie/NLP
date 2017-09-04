import ngram_utils

import numpy as np
import unittest

class DummyLM(object):
    """Trivial unigram model."""
    def __init__(self, word_map, context_size=0):
        self.unigram_model = word_map
        self.words = [w for w in self.unigram_model.keys()]
        self.order_n = context_size + 1
    
    def next_word_proba(self, word, seq):
        return self.unigram_model.get(word, 0.0)


class TestNGramUtils(unittest.TestCase):
    def test_predict_next(self):
        lm = DummyLM({'hello': 0.0, 'world': 1.0})
        self.assertEqual('world', ngram_utils.predict_next(lm, None))

    def test_score_seq(self):
        lm = DummyLM({
            'hello': 0.1,
            'world': 0.2,
            'how': '0.7'
        })

        self.assertAlmostEqual(
                (np.log2(0.1) + np.log2(0.2), 2),
                ngram_utils.score_seq(lm, ['hello', 'world']))

        self.assertAlmostEqual(
                (np.log2(0.1) + np.log2(0.2), 2),
                ngram_utils.score_seq(lm, ['<s>', 'hello', 'world', '</s>']))
        
    def test_score_seq_with_context(self):
        lm = DummyLM({
            'hello': 0.1,
            'world': 0.2,
            'how': '0.7'
        }, context_size=1)

        self.assertAlmostEqual(
                (np.log2(0.2), 1),
                ngram_utils.score_seq(lm, ['hello', 'world']))

        self.assertAlmostEqual(
                (np.log2(0.1) + np.log2(0.2), 2),
                ngram_utils.score_seq(lm, ['<s>', 'hello', 'world', '</s>']))
 
if __name__ == '__main__':
    unittest.main()
