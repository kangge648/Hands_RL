import numpy as np

nrow = 10
ncol = 4
n_action = 4

Q_table = np.zeros([nrow * ncol, n_action])
Q_table[10][1] = 1
action = np.argmax(Q_table[10])
print(type(Q_table[10]))