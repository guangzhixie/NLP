import numpy as np

def predict_next(lm, seq, **kw):
    """Sample a word from the conditional distribution."""
    probs = [lm.next_word_proba(word, seq, **kw) for word in lm.words]
    return np.random.choice(lm.words, p=probs)


def score_seq(lm, seq, verbose=False):
    """Compute log probability (base 2) of the given sequence."""
    context_size = lm.order_n - 1
    score = 0.0
    count = 0
    # Start at third word, since we need a full context.
    for i in range(context_size, len(seq)):
        if (seq[i] == "<s>" or seq[i] == "</s>"):
            continue  # Don't count special tokens in score.
        context = seq[i-context_size:i]
        s = np.log2(lm.next_word_proba(seq[i], context))
        score += s
        count += 1
        # DEBUG.
        if verbose:
            print "log P(%s | %s) = %.03f" % (seq[i], " ".join(context), s)
    return score, count


def print_stats(lm):
    """Output summary statistics about our language model."""
    print "=== N-gram Language Model stats ==="
    for i in range(lm.order_n):
        unique_ngrams = sum(len(c) for k,c in lm.counts.iteritems()
                if len(k) == i)
        print "%g unique %d-grams" % (unique_ngrams, i+1)

    optimal_memory_bytes = sum(
            (4 * len(k) + 20 * len(v))
             for k, v in lm.counts.iteritems())
    print ("Optimal memory usage (counts only): %d MB" %
            (optimal_memory_bytes / (2**20)))
