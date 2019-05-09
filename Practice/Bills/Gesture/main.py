import numpy as np
SEED = 1234
DIRECTOR = 'Practice/Bills/Gesture/'
np.random.seed(SEED)
data = np.loadtxt(DIRECTOR+'EMG.csv', skiprows=1, delimiter=',', dtype=np.int)
print(1)
