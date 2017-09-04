
import collections
import itertools
import logging
logger = logging.getLogger(__name__)

import numpy as np

import pcfg

from nltk.tree import ProbabilisticTree

def lookup_with_key_cast(d, key, default, cast_fn):
    """Like dict.get(...), but applies cast_fn to keys before matching.

    Useful if actual keys are a wrapper type like nltk.grammar.Nonterminal, but
    you only care about matching the string form.
    """
    cast_to_keys = {cast_fn(k):k for k in d}
    if key in cast_to_keys:
        return d[cast_to_keys[key]]
    return default

def make_chart():
    """Create an empty chart."""
    dummy_tree_factory = lambda: ProbabilisticTree('', [], logprob=-np.inf)
    cell_factory = lambda: collections.defaultdict(dummy_tree_factory)
    return collections.defaultdict(cell_factory)

def ordered_spans(n):
    """Generate all spans, sorted bottom-up.

    Returns a list all spans [i,j) where 0 <= i < j <= n, sorted by the length
    of the span (j - i).

    For example, ordered_spans(4) would return spans:
        [0, 1) [1, 2) [2, 3) [3, 4)
        [0, 2) [1, 3) [2, 4)
        [0, 3) [1, 4)
        [0, 4)
    """
    key_fn = lambda (a, b): b - a
    return sorted([(i,j) for i in xrange(n)
                         for j in xrange(i+1, n+1)], key=key_fn)

class CKYParser(object):

    def __init__(self, grammar, beam_size=None, beam_width=None):
        """Construct a CKY parser.

        Args:
            grammar: (pcfg.PCFG) weighted CFG
            beam_size: maximum cell population for chart pruning
            beam_width: beam width for chart pruning
        """
        assert(isinstance(grammar, pcfg.PCFG))
        assert(grammar.parsing_index is not None)
        self._grammar = grammar

        assert(beam_size >= 1 or beam_size is None)
        assert(isinstance(beam_size, int) or beam_size is None)
        assert(beam_width > 0 or beam_width is None)
        self._beam_size = beam_size
        self._beam_width = beam_width

    def check_vocabulary(self, words):
        """Check tokens for OOV words.

        Args:
            words: list(string) words to check

        Returns:
            list(int) indices (into words) of OOV words
        """
        return [i for i,w in enumerate(words)
                if not self._grammar.lookup_rhs(w)]

    def _apply_preterminal_rules(self, words, chart):
        """Populate the bottom row of the CKY chart.

           Specifically, apply preterminal unary rules to go from word to preterminal.

           Args:
             - words: sequence of words to parse
             - chart: the chart to populate

           Returns: False if a preterminal could not be found in the grammar for a word.
                    True otherwise.
        """
        
        # Handle preterminal rules A -> a
        # For the ith token, you should populate cell (i,i+1).
        for i, word in enumerate(words):
            cell_key = (i,i+1)
            pass

            if (word,) not in self._grammar.parsing_index:
                return False
            
            for t in self._grammar.parsing_index[(word,)]:
                pos_tag = t[0]
                score = t[1]
                chart[cell_key][pos_tag] = ProbabilisticTree(pos_tag, [word], logprob=score)

        return True


    def _apply_binary_rules(self, N, chart):
        """Populate the remainder of the chart, assuming the bottom row is complete.

           Iterating throught the chart from the bottom up, apply all available
           binary rules at each position in the chart.  Each cell of the chart should
           enumerate the heads that can be produced there and the score corresponding
           to their most efficient construction.

           Args:
             - N: the number of words
             - chart: the chart to populate, see _apply_preterminal_rules for a detailed description.
        """
        
        # Iterate through the chart, handling nonterminal rules A -> B C
        # Use the ordered_spans function to get a list of spans from the bottom up.
        for (i, j) in ordered_spans(N):
            for split in xrange(i+1, j):
                # Consider all possible A -> B C
                pass
            
                for lhs_tree in chart[(i, split)].values():
                    for rhs_tree in chart[(split, j)].values():
                        B = lhs_tree.label()
                        C = rhs_tree.label()
                        
                        for (A, score) in self._grammar.lookup_rhs(B, C):
                    
                            total_score = lhs_tree.logprob() + rhs_tree.logprob() + score
                            
                            if total_score > chart[(i, j)][A].logprob():
                                chart[(i, j)][A] = ProbabilisticTree(A, [lhs_tree, rhs_tree], logprob=total_score)



    def parse(self, words, target_type=None, return_chart=False):
        """Run the CKY chart-parsing algorithm.

        Given a sequence of words and a weighted context-free grammar, finds the
        most likely derivation Tree.

        Args:
            words: list(string) sentence to parse
            target_type: (string OR nltk.grammar.Nonterminal) if specified, will
                return the highest scoring derivation of that type (e.g. 'S'). If
                None, will return the highest scoring derivation of any type.
            return_chart: if true, will also return the full chart.

        Returns:
            (nltk.tree.ProbabilisticTree): optimal derivation
        """
        # The chart is a map from span -> symbol -> derivation
        # Formally: tuple(int, int) -> nltk.grammar.Nonterminal
        #                                  -> nltk.tree.ProbabilisticTree
        # The defaultdict machinery (see Assignment 2, Part 1) will populate cells
        # with dummy entries so you don't need to handle the special case of
        # an empty cell in your inner loop.
        # (See make_chart earlier in this file if you want to see this in action.)
        chart = make_chart()

        N = len(words)

        # Words -> preterminals via unary rules.
        # (i.e. populate the bottom row of the chart)
        if not self._apply_preterminal_rules(words, chart):
            logger.warn("Empty parse - apply_preterminal_rules failed."
                        + " This is usually due to out-of-vocabulary words.")
            oov_idxs = self.check_vocabulary(words)
            for idx in oov_idxs:
                logger.warn("Out-of-vocabulary word [%d]: '%s'", idx, words[idx])
            return (None, chart) if return_chart else None

        # Populate the rest of the chart from binary rules.
        self._apply_binary_rules(N, chart)

        # Verify we were able to produce something in the cell spanning the
        # entire sentence.  If we can't, the sentence isn't parseable under
        # our grammar.
        if len(chart[(0,N)]) == 0:
            logger.warn("Empty parse - no derivation in top cell [0,%d)." % N)
            return (None, chart) if return_chart else None

        # Final parse tree is just the best-scoring derivation in the top cell.
        # (If a specific target_type is requested - often 'S' - return that parse
        # tree even if there is a higher scoring parse available.)
        if target_type is None:
            ret = max(chart[(0,N)].values(), key=lambda t: t.logprob())
        else:
            # Try for both string match and coercion
            ret = chart[(0,N)].get(target_type, None)
            if ret is None:
                #  ret = chart[(0,N)].get(nltk.grammar.Nonterminal(target_type), None)
                ret = lookup_with_key_cast(chart[(0,N)], target_type,
                                           None, str)
        if ret is None:
            logger.warn("Empty parse - target type '%s' not found in top cell [0,%d)." % (target_type, N))
        return (ret, chart) if return_chart else ret

