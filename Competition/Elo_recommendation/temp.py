import pandas as pd
import numpy as np
np.random.seed(0)
file1 = pd.DataFrame(np.random.randn(3, 2), index=['1', '2', '3'], columns=['a', 'b'])
# file2 = pd.DataFrame({'A': pd.Timestamp('20190824'), 'B': pd.Series(1, index=[1, 2, 3]), 'C': pd.Series(1, index=[0, 1, 2, 3])})

