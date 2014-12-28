from vbench.benchmark import Benchmark
from datetime import datetime

common_setup = """from pandas_vb_common import *
"""

#----------------------------------------------------------------------

setup = common_setup + """
from pandas.core.sparse import SparseSeries, SparseDataFrame

K = 50
N = 50000
rng = np.asarray(date_range('1/1/2000', periods=N,
                           freq='T'))

# rng2 = np.asarray(rng).astype('M8[ns]').astype('i8')

series = {}
for i in range(1, K + 1):
    data = np.random.randn(N)[:-i]
    this_rng = rng[:-i]
    data[100:] = np.nan
    series[i] = SparseSeries(data, index=this_rng)
"""
stmt = "SparseDataFrame(series)"

bm_sparse1 = Benchmark(stmt, setup, name="sparse_series_to_frame",
                       start_date=datetime(2011, 6, 1))


setup = common_setup + """
from pandas.core.sparse import SparseDataFrame
"""

stmt = "SparseDataFrame(columns=np.arange(100), index=np.arange(1000))"

sparse_constructor = Benchmark(stmt, setup, name="sparse_frame_constructor",
                               start_date=datetime(2012, 6, 1))


setup = common_setup + """
s = pd.Series([3.0, nan, 1.0, 2.0, nan, nan])
s.index = pd.MultiIndex.from_tuples([(1, 2, 'a', 0),
                                     (1, 2, 'a', 1),
                                     (1, 1, 'b', 0),
                                     (1, 1, 'b', 1),
                                     (2, 1, 'b', 0),
                                     (2, 1, 'b', 1)])
# SparseSeries
ss = s.to_sparse()
"""

stmt = "ss.to_coo(ilevels=[0, 1], jlevels=[2, 3], sort_labels=True)"

sparse_series_to_coo = Benchmark(stmt, setup, name="sparse_series_to_coo", start_date=datetime(2014, 12, 28))
