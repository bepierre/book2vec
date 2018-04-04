import numpy as np
from gensim.models.doc2vec import Doc2Vec
import glob
from scipy import spatial


vec_names = np.load('../models/vec_names_20k.npy').tolist()
book_filenames = sorted(glob.glob('../data/BookCorpus/*/*txt'))

load = False

if not load:
    model = Doc2Vec.load('../models/par2vec_300_20k.doc2vec')

    norm_vecs = []

    for b in book_filenames:
        b = b[0:-4]
        b_vecs = []
        for vec_name in vec_names:
            if b in vec_name:
                b_vecs.append(model[vec_names.index(vec_name)])

        b_mean = np.mean(b_vecs, axis=0)
        b_var = np.var(b_vecs, axis=0)
        for vec_name in vec_names:
            if b in vec_name:
                #print(vec_name)
                norm_vecs.append((model[vec_names.index(vec_name)] - b_mean) / b_var)

    np.save('../models/norm_par_vecs.npy', norm_vecs)
else:
    norm_vecs = np.load('../models/norm_par_vecs.npy')

#model = Doc2Vec.load('../models/par2vec_300_20k.doc2vec')
#norm_vecs = model.docvecs.vectors_docs

tree = spatial.KDTree(norm_vecs)

#b = '../data/BookCorpus/Thriller/James_Bond-1'
#b = '../data/BookCorpus/Thriller/Da_Vinci_Code'
#b = '../data/BookCorpus/Romance/Attachments'
#b = '../data/BookCorpus/Fantasy/Mistborn-1'
b = '../data/BookCorpus/Science_fiction/Asimov47'


p = 1

distances, closest = tree.query(norm_vecs[vec_names.index(b + '_par_' + str(p))], k= 10)

i = 0
for c in closest:
    print(vec_names[c])
    print(distances[i])
    i += 1