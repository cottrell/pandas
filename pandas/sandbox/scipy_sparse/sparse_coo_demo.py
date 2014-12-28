from pandas import *
from numpy import nan

s = pandas.Series([3.0, nan, 1.0, 3.0, nan, nan])
s.index = pandas.MultiIndex.from_tuples([(1, 2, 'a', 0),
                                         (1, 2, 'a', 1),
                                         (1, 1, 'b', 0),
                                         (1, 1, 'b', 1),
                                         (2, 1, 'b', 0),
                                         (2, 1, 'b', 1)])

# SparseSeries
ss = s.to_sparse()

A, il, jl = ss.to_coo(ilevels=[0, 1], jlevels=[2, 3], sort_labels=True)

print(ss)
print(A)
print(il)
print(jl)

import IPython
IPython.embed()
