import glob
from gensim.models.doc2vec import Doc2Vec
import numpy as np

book_filenames1 = sorted(glob.glob('../data/BookCorpus/*/*txt'))

model = Doc2Vec.load('../models/book2vec_300.doc2vec')

book_vecs = np.load('../models/b2p2v_book_vecs_c.npy')
book_filenames2 = np.load('../models/b2p2v_book_filenames.npy')

print(book_filenames1[0])
print(model[0])

print(book_filenames2[0])
print(book_vecs[0])