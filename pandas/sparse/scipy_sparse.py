"""
Interaction with scipy.sparse matrices.

Currently only includes SparseSeries.to_coo helpers.


TODO:
    * more tests
    * from_coo
    * vbench
"""
from pandas.core.frame import DataFrame
from pandas.core.index import MultiIndex
from pandas.core.series import Series
import itertools
import numpy
import scipy.sparse

def _check_partition(left, right, whole):
    left = set(left)
    right = set(right)
    whole = set(whole)
    assert(len(left.intersection(right))==0)
    assert(left.union(right) == whole)

def _get_index_level_subset(s, subset):
    return(list(zip(*[s.index.get_level_values(i) for i in subset])))

def _squish(s):
    seen = set()
    out = [tuple(x) for x in s]
    out = [x for x in out if x not in seen and not seen.add(x)]
    return(out)

def _get_label_to_i_dict(labels, sorted=False):
    labels = _squish(labels)
    if sorted:
        labels = sorted(list(labels))
    d = dict({k: i for i, k in enumerate(labels)})
    return(d)

def _get_sparse_coords(ss, blocs, blength, levels):
    il = _get_index_level_subset(ss, levels)
    sparse_labels = list(itertools.chain(*[il[i:(i+j)] for i, j in zip(blocs, blength)]))
    idict = _get_label_to_i_dict(sparse_labels)
    i = [idict[tuple(k)] for k in sparse_labels]
    inv_dict = {v: k for k, v in idict.items()}
    ordered_labels = [inv_dict[k] for k in range(len(idict))]
    return(i, ordered_labels)

def _to_ijv(ss, ilevels=(0,), jlevels=(1,), sort_labels=False):
    """ For arbitrary (MultiIndexed) SparseSeries return (v, i, j, ilabels, jlabels) where (v, (i, j)) is suitable for
    passing to scipy.sparse.coo constructory. """
    # index and column levels must be a partition of the index
    _check_partition(ilevels, jlevels, range(ss.index.nlevels))
    v = ss._data.values._valid_sp_values
    blocs = ss._data.values.sp_index.blocs
    blength = ss._data.values.sp_index.blengths
    i, il = _get_sparse_coords(ss, blocs, blength, ilevels)
    j, jl = _get_sparse_coords(ss, blocs, blength, jlevels)
    return(v, i, j, il, jl)

def sparse_series_to_coo(ss, ilevels=(0,), jlevels=(1,), sort_labels=False):
    """ Convert a SparseSeries to a scipy.sparse.coo_matrix using ilevels, jlevels as the row, column labels.
    Returns the sparse_matrix as well as row and column labels. """
    # TODO: how handle duplicate index entries? what kind of error to throw?
    assert(~any(ss.index.to_series().duplicated()))
    v, i, j, il, jl = _to_ijv(ss, ilevels=ilevels, jlevels=jlevels, sort_labels=sort_labels)
    sparse_matrix = scipy.sparse.coo_matrix((v, (i, j)), shape=(len(il), len(jl)))
    return(sparse_matrix, il, jl)
