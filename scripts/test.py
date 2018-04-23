import numpy as np
from gensim.models.doc2vec import Doc2Vec
import glob
import scipy.io

size = '_full'
vec_size = 300
#par_length = 1000
words_per_par = 400

book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))
vec_names = np.load('../models/vec_names'+size +'_'+ str(int(words_per_par / 100)) + 'c_w.npy').tolist()
genre_names = sorted(glob.glob('../data/BookCorpusFull/*'))


print(vec_names[0:1])

# list2 = np.array(book_filenames, dtype=np.object)
# scipy.io.savemat('../models/book_names.mat', mdict={'book_names':list2})