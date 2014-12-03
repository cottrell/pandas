from pandas import *
import numpy
from numpy.random import randn

a = numpy.arange(10 * 4)
a.shape = (10, 4)
df = DataFrame(a, columns=['a', 'b', 'c', 'd'])
df.iloc[3:-2,] = np.nan
df.iloc[:3,2:] = np.nan
df.iloc[-2:,:2] = np.nan
df.columns = pandas.MultiIndex.from_tuples([(1, 2, 'a'), (1, 1, 'b'), (2, 1, 'b'), (2, 2, 'c')]).T

# SparseDataFrame
sdf = df.to_sparse()

# SparseSeries
ss = df.unstack().to_sparse()

A, il, jl = ss.to_coo(ilevels=[0, 1], jlevels=[2, 3])

print(ss)
print(A)
print(il)
print(jl)

# import IPython
# IPython.embed()
