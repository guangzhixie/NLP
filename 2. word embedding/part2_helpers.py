import numpy as np
import scipy.sparse

from shared_lib import vocabulary
from shared_lib import utils

# Version check for sklearn
import sklearn
assert(sklearn.__version__ >= "0.18")

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, ShuffleSplit, GridSearchCV

##
# Helper functions for part 2 of assignment 2.
# These are the same functions we defined in the week 3 demo notebook
# (materials/week3/embeddings.ipynb); we're defining them in a separate file
# here for to avoid cluttering the notebook.

def sents_to_tokens(sents, vocab):
    """Returns an flattened list of the words in the sentences, with normal padding."""
    padded_sentences = (["<s>"] + s + ["</s>"] for s in sents)
    # This will canonicalize words, and replace anything not in vocab with <unk>
    return np.array([utils.canonicalize_word(w, wordset=vocab.wordset)
                     for w in utils.flatten(padded_sentences)], dtype=object)

def cooccurrence_matrix(token_ids, V, K=2):
    # We'll use this as an "accumulator" matrix
    C = scipy.sparse.csc_matrix((V,V), dtype=np.float32)

    for k in range(1, K+1):
        print u"Counting pairs (i, i \u00B1 %d) ..." % k
        i = token_ids[:-k]  # current word
        j = token_ids[k:]   # k words ahead
        data = (np.ones_like(i), (i,j))  # values, indices
        Ck_plus = scipy.sparse.coo_matrix(data, shape=C.shape, dtype=np.float32)
        Ck_plus = scipy.sparse.csc_matrix(Ck_plus)
        Ck_minus = Ck_plus.T  # Consider k words behind
        C += Ck_plus + Ck_minus

    print "Co-occurrence matrix: %d words x %d words" % (C.shape)
    print "  %.02g nonzero elements" % (C.nnz)
    return C

def PPMI(C):
    """Tranform a counts matrix to PPMI.

    Args:
      C: scipy.sparse.csc_matrix of counts C_ij

    Returns:
      (scipy.sparse.csc_matrix) PPMI(C) as defined above
    """
    Z = float(C.sum())  # total counts
    # sum each column (along rows)
    Zc = np.array(C.sum(axis=0), dtype=np.float64).flatten()
    # sum each row (along columns)
    Zr = np.array(C.sum(axis=1), dtype=np.float64).flatten()

    # Get indices of relevant elements
    ii, jj = C.nonzero()  # row, column indices
    Cij = np.array(C[ii,jj], dtype=np.float64).flatten()

    ##
    # PMI equation
    pmi = np.log(Cij * Z / (Zr[ii] * Zc[jj]))
    ##
    # Truncate to positive only
    ppmi = np.maximum(0, pmi)  # take positive only

    # Re-format as sparse matrix
    ret = scipy.sparse.csc_matrix((ppmi, (ii,jj)), shape=C.shape,
                                  dtype=np.float64)
    ret.eliminate_zeros()  # remove zeros
    return ret

def SVD(X, d=100):
    """Returns word vectors from SVD.

    Args:
      X: m x n matrix
      d: word vector dimension

    Returns:
      Wv : m x d matrix, each row is a word vector.
    """
    transformer = TruncatedSVD(n_components=d, random_state=1)
    Wv = transformer.fit_transform(X)
    # Normalize to unit length
    Wv = Wv / np.linalg.norm(Wv, axis=1).reshape([-1,1])
    return Wv, transformer.explained_variance_


def train_logistic_cv(X, y, N=None, verbose=False):
    param_grid = [
	{'C': [1, 1000], 'penalty': ['l1']}
    ]

    n_splits = 10
    cv = ShuffleSplit(n_splits=n_splits, train_size=N, random_state=42)
    model = LogisticRegression(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=cv,
                               verbose=int(verbose))
    grid_search.fit(X, y)

    score = grid_search.best_score_

    ## Find std of score
    rts = grid_search.cv_results_['rank_test_score']
    mts = grid_search.cv_results_['mean_test_score']
    sts = grid_search.cv_results_['std_test_score']
    ii = np.argmin(rts)  # get index of best score
    assert(mts[ii] == score)
    score, std = mts[ii], sts[ii]/np.sqrt(n_splits - 1.0)

    if verbose:
        print "Best params: " + str(grid_search.best_params_)
        print "Best dev accuracy: {0:.02f}% +/- {1:.02f}%".format(100*score,
                                                                  100*std)
        print ""

    return score, std
