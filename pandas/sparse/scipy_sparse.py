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


def _check_is_partition(parts, whole):
    whole = set(whole)
    parts = [set(x) for x in parts]
    if set.intersection(*parts) != set():
        raise(ValueError('Is not a partition because intersection is not null.'))
    if set.union(*parts) != whole:
        raise(ValueError('Is not a partition becuase union is not the whole.'))


def _get_index_level_subset(s, subset):
    return(list(zip(*[s.index.get_level_values(i) for i in subset])))


def _squish(s):
    seen = set()
    out = [tuple(x) for x in s]
    out = [x for x in out if x not in seen and not seen.add(x)]
    return(out)


def _get_label_to_i_dict(labels, sort_labels=False):
    labels = _squish(labels)
    if sort_labels:
        labels = sorted(list(labels))
    d = dict({k: i for i, k in enumerate(labels)})
    return(d)


def _get_sparse_coords(ss, levels, sort_labels=False):
    blocs = ss._data.values.sp_index.blocs
    blength = ss._data.values.sp_index.blengths
    il = _get_index_level_subset(ss, levels)

    # TODO: find a better/clearer way to do this part
    idict = _get_label_to_i_dict(il, sort_labels=sort_labels)
    inv_dict = {v: k for k, v in idict.items()}

    nonnull_labels = list(itertools.chain(*[il[i:(i + j)] for i, j in zip(blocs, blength)]))
    ind = [idict[tuple(k)] for k in nonnull_labels]
    ordered_labels = [inv_dict[i] for i in range(len(idict))]

    return(ind, ordered_labels)


def _to_ijv(ss, ilevels=(0,), jlevels=(1,), sort_labels=False):
    """ For arbitrary (MultiIndexed) SparseSeries return (v, i, j, ilabels, jlabels) where (v, (i, j)) is suitable for
    passing to scipy.sparse.coo constructory. """
    # index and column levels must be a partition of the index
    _check_is_partition([ilevels, jlevels], range(ss.index.nlevels))
    v = ss._data.values._valid_sp_values
    i, il = _get_sparse_coords(ss, ilevels, sort_labels=sort_labels)
    j, jl = _get_sparse_coords(ss, jlevels, sort_labels=sort_labels)
    return(v, i, j, il, jl)


def sparse_series_to_coo(ss, ilevels=(0,), jlevels=(1,), sort_labels=False):
    """ Convert a SparseSeries to a scipy.sparse.coo_matrix using ilevels, jlevels as the row, column labels.
    Returns the sparse_matrix as well as row and column labels. """
    # TODO: how handle duplicate index entries? what kind of error to throw?
    assert(~any(ss.index.to_series().duplicated()))
    v, i, j, il, jl = _to_ijv(ss, ilevels=ilevels, jlevels=jlevels, sort_labels=sort_labels)
    sparse_matrix = scipy.sparse.coo_matrix((v, (i, j)), shape=(len(il), len(jl)))
    return(sparse_matrix, il, jl)
