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
from pandas.compat import OrderedDict

def _squish(s):
    """ Uniquify s while preserving order. Elements of s must be appendable to set. """
    seen = set()
    return([x for x in s if x not in seen and not seen.add(x)])

def _get_label_to_i_dict(labels, sort_labels=False):
    """ Return OrderedDict of unique labels to number. Optionally sort by label. """
    labels = _squish(map(tuple, labels))
    if sort_labels:
        labels = sorted(list(labels))
    d = OrderedDict((k, i) for i, k in enumerate(labels))
    return(d)

def _get_index_subset_to_coord_dict(index, subset, sort_labels=False):
    ilabels = list(zip(*[index.get_level_values(i) for i in subset]))
    labels_to_i = _get_label_to_i_dict(ilabels, sort_labels=sort_labels)
    return(labels_to_i)

def _check_is_partition(parts, whole):
    whole = set(whole)
    parts = [set(x) for x in parts]
    if set.intersection(*parts) != set():
        raise(ValueError('Is not a partition because intersection is not null.'))
    if set.union(*parts) != whole:
        raise(ValueError('Is not a partition becuase union is not the whole.'))

def _to_ijv(ss, ilevels=(0,), jlevels=(1,), sort_labels=False):
    """ For arbitrary (MultiIndexed) SparseSeries return (v, i, j, ilabels, jlabels) where (v, (i, j)) is suitable for
    passing to scipy.sparse.coo constructor. """
    # index and column levels must be a partition of the index
    _check_is_partition([ilevels, jlevels], range(ss.index.nlevels))

    # from the SparseSeries: get the labels and data for non-null entries
    values = ss._data.values._valid_sp_values
    blocs = ss._data.values.sp_index.blocs
    blength = ss._data.values.sp_index.blengths
    nonnull_labels = list(itertools.chain(*[ss.index.values[i:(i + j)] for i, j in zip(blocs, blength)]))

    def get_indexers(levels):
        """ Return sparse coords and dense labels for subset levels """
        values_ilabels = [tuple(x[i] for i in levels) for x in nonnull_labels]
        labels_to_i = _get_index_subset_to_coord_dict(ss.index, levels, sort_labels=sort_labels)
        i_coord = [labels_to_i[i] for i in values_ilabels]
        return(i_coord, list(labels_to_i.keys()))

    i_coord, i_labels = get_indexers(ilevels)
    j_coord, j_labels = get_indexers(jlevels)

    return(values, i_coord, j_coord, i_labels, j_labels)

def sparse_series_to_coo(ss, ilevels=(0,), jlevels=(1,), sort_labels=False):
    """ Convert a SparseSeries to a scipy.sparse.coo_matrix using index levels ilevels, jlevels as the row and column
    labels respectively. Returns the sparse_matrix, row and column labels. """
    # TODO: how handle duplicate index entries? what kind of error to throw?
    if any(ss.index.to_series().duplicated()):
        raise(ValueError('Duplicate index entries are not allowed in to_coo transformation.'))
    v, i, j, il, jl = _to_ijv(ss, ilevels=ilevels, jlevels=jlevels, sort_labels=sort_labels)
    sparse_matrix = scipy.sparse.coo_matrix((v, (i, j)), shape=(len(il), len(jl)))
    return(sparse_matrix, il, jl)
