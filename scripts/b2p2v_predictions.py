import numpy as np
import glob
from gensim.models.doc2vec import Doc2Vec
from scipy import spatial



vec_names = np.load('../models/vec_names_20k.npy').tolist()
book_filenames = sorted(glob.glob('../data/BookCorpus/*/*txt'))
model = Doc2Vec.load('../models/par2vec_300_20k.doc2vec')
par_vecs = model.docvecs.vectors_docs

tree = spatial.KDTree(par_vecs)


cleaned_vecs = np.load('../models/book_par_vecs_20k.npy')
predicted_vecs = np.load('../models/b2p2v_predicted_vecs.npy')

p = 1

distances, closest = tree.query(predicted_vecs[100][10], k= 10)
a = np.atleast_2d(par_vecs)
print("%d, %d" % (np.min(a), np.max(a)))
i = 0
for c in closest:
    print(vec_names[c])
    print(distances[i])
    i += 1