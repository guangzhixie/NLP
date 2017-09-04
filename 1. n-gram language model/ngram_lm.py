# An elegant weapon, for a more civilized age
from __future__ import division

from collections import defaultdict
import numpy as np

class AddKTrigramLM(object):
    """Trigram LM with add-k smoothing."""
    order_n = 3

    def __eq__(self, other):
        state_vars = ['k', 'counts', 'context_totals', 'words', 'V']
        return all([getattr(self, v) == getattr(other, v) for v in state_vars])

    def __init__(self, tokens):
        """Build smoothed trigram model.

        We _don't_ want to actually normalize the probabilities at training time. 
        Instead, we'll compute the corpus counts C_abc = C(w_2, w_1, w) 
        and C_ab = C(w_2, w_1), after which we can compute the probabilities 
        on the fly for any value of k. (We'll do this
        in the next_word_proba() function.)

        Args:
          tokens: (list or np.array) of training tokens

        Returns:
          None
        """
        self.k = 0.0
        # Raw trigram counts over the corpus.
        # c(w | w_1 w_2) = self.counts[(w_2,w_1)][w]
        # Be sure to use tuples (w_2,w_1) as keys, *not* lists [w_2,w_1]
        self.counts = defaultdict(lambda: defaultdict(lambda: 0.0))

        # Map of (w_1, w_2) -> int
        # Entries are c( w_2, w_1 ) = sum_w c(w_2, w_1, w)
        self.context_totals = dict()

        # Track unique words seen, for normalization
        # Use wordset.add(word) to add words
        wordset = set()

        # Iterate through the word stream once
        # Compute trigram counts as in SimpleTrigramLM
        w_1, w_2 = None, None
        for word in tokens:
            wordset.add(word)
            if w_1 is not None and w_2 is not None:
                self.counts[(w_2,w_1)][word] += 1
            # Update context
            w_2 = w_1
            w_1 = word

        # Compute context counts
        for context, ctr in self.counts.iteritems():
            self.context_totals[context] = sum(ctr.itervalues())

        # Total vocabulary size, for normalization
        self.words = list(wordset)
        self.V = len(self.words)

    def set_live_params(self, k=0.0, **params):
        self.k = k

    def next_word_proba(self, word, seq):
        """Next word probability for smoothed n-gram.

        Args:
          word: (string) w in P(w | w_1 w_2 )
          seq: (list of string) [w_1, w_2, w_3, ...]

        Returns:
          (float) P_k(w | w_1 w_2), according to the model
        """
        context = tuple(seq[-2:])  # (w_2, w_1)
        k = self.k
        
        c_abc = self.counts.get(context, {}).get(word, 0)
        c_ab = self.context_totals.get(context, 0)
        return float(c_abc + k)/(c_ab + k*self.V)



class KNTrigramLM(object):
    """Trigram LM with Kneser-Ney smoothing."""
    order_n = 3

    def __eq__(self, other):
        state_vars = ['delta', 'counts', 'type_contexts',
                      'context_totals', 'context_nnz', 'type_fertility',
                      'z_tf', 'words']
        return all([getattr(self, v) == getattr(other, v) for v in state_vars])

    def __init__(self, tokens):
        """Build smoothed trigram model.

        The usual identification of variables:
          w : c : current word
          w_1 : w_{i-1} : b : previous word
          w_2 : w_{i-2} : a : previous-previous word

        Args:
          tokens: (list or np.array) of training tokens

        Returns:
          None
        """
        self.delta = 0.75
        # Raw counts over the corpus.
        # Keys are context (N-1)-grams, values are dicts of word -> count.
        # Can access C(w | w_{i-1}, ...) as:
        # unigram: self.counts[()][w]
        # bigram:  self.counts[(w_1,)][w]
        # trigram: self.counts[(w_2,w_1)][w]
        self.counts = defaultdict(lambda: defaultdict(lambda: 0))
        # As in AddKTrigramLM, but also store the unigram and bigram counts
        # self.context_totals[()] = (total word count)
        # self.context_totals[(w_1,)] = c(w_1)
        # self.context_totals[(w_2, w_1)] = c(w_2, w_1)
        self.context_totals = dict()
        # Also store in self.context_nnz the number of nonzero entries for each
        # context; as long as \delta < 1 this is equal to nnz(context) as
        # defined in the notebook.
        self.context_nnz = dict()

        # Context types: store the set of preceding words for each word
        # map word -> {preceding_types}
        self.type_contexts = defaultdict(lambda: set())
        # Type fertility is the size of the set above
        # map word -> |preceding_types|
        self.type_fertility = dict()
        # z_tf is the sum of type fertilities
        self.z_tf = 0.0


        # Iterate through the word stream once
        # Compute unigram, bigram, trigram counts and type fertilities
        w_1, w_2 = None, None
        for word in tokens:
        
            # unigram
            self.counts[()][word] += 1
            
            # bigram
            if w_1 is not None:
                self.counts[(w_1,)][word] += 1
            
            # trigram
            if w_1 is not None and w_2 is not None:
                self.counts[(w_2,w_1)][word] += 1
                
            # type fertilities
            if w_1 is not None:
                self.type_contexts[word].add(w_1)

            
            # Update context
            w_2 = w_1
            w_1 = word

        ##
        # Compute type fertilities and normalization constants now,
        # but not actually store the normalized probabilities. That way, we can compute
        # them (efficiently) on the fly.

        # Count the total for each context.
        for context, ctr in self.counts.iteritems():
            self.context_totals[context] = sum(ctr.itervalues())

        # Count the number of nonzero entries for each context.
            self.context_nnz[context] = len(ctr)

        # Compute type fertilities, and the sum z_tf.
        for word, preceding_types in self.type_contexts.iteritems():
            self.type_fertility[word] = len(preceding_types)

        self.z_tf = float(sum(self.type_fertility.values()))

        # Total vocabulary size, for normalization
        self.words = self.counts[()].keys()
        self.V = len(self.words)

    def set_live_params(self, delta = 0.75, **params):
        self.delta = delta

    def kn_interp(self, word, context, delta, pw):
        """Compute KN estimate P_kn(w | context) given a backoff probability

        Args:
          word: (string) w in P(w | context )
          context: (tuple of string)
          delta: (float) discounting term
          pw: (float) backoff P_kn(w | less_context), precomputed

        Returns:
          (float) P_kn(w | context)
        """
        
        c_ab = self.context_totals.get(context, 0)
        if c_ab == 0:
            return pw
        
        c_abc = self.counts.get(context, {}).get(word, 0)
        alpha_ab = delta * self.context_nnz.get(context, 0)/c_ab
        
        abs_discount = c_abc - delta if (c_abc - delta) > 0 else 0
        
        return abs_discount/c_ab + alpha_ab*pw


    def next_word_proba(self, word, seq):
        """Compute next word probability with KN backoff smoothing.

        Args:
          word: (string) w in P(w | w_1 w_2 )
          seq: (list of string) [w_1, w_2, w_3, ...]
          delta: (float) discounting term

        Returns:
          (float) P_kn(w | w_1 w_2)
        """
        delta = delta = self.delta
        # KN unigram, then recursively compute bigram, trigram
        pw1 = self.type_fertility[word] / self.z_tf
        pw2 = self.kn_interp(word, tuple(seq[-1:]), delta, pw1)
        pw3 = self.kn_interp(word, tuple(seq[-2:]), delta, pw2)
        return pw3
