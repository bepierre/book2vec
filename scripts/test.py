import numpy as np

data = np.load('../models/book_par_vecs_20k.npy')#
print(data[0][0]-data[0][2])
print(np.sum((data[0][0]-data[0][2])**2))