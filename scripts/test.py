import numpy as np

data = np.load('../models/book_par_vecs_20k.npy')#
print(np.sum((data[0]-data[1])**2))