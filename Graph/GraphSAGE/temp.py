import gp_loaddata
import gp_minibatch
import numpy as np
'''
root = r'Datasets\Temp\Graphsage_data\toy-ppi-G.json'
graph, label, feature = gp_loaddata.load_data(root)
test_edge_itor = gp_minibatch.EdgeMinibatchIterator(graph)
'''
a = [[1, 2, 3]]
b = [[4, 5, 6]]
c = np.concatenate((a, b), axis=0)
d = np.stack((a, b), axis=1)
print(c)
