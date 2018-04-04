import numpy as np
from gensim.models.doc2vec import Doc2Vec
import glob


vec_names = np.load('../models/vec_names_20k.npy').tolist()
book_filenames = sorted(glob.glob('../data/BookCorpus/*/*txt'))

model = Doc2Vec.load('../models/par2vec_300_20k.doc2vec')

#b = '../data/BookCorpus/Fantasy/The_Hobbit'
#b = '../data/BookCorpus/Thriller/Da_Vinci_Code'
b = '../data/BookCorpus/Horror/Bullet'

print(model.docvecs.most_similar(positive = [model[vec_names.index(b + '_par_1')]]))

'''
b_vecs = []
for vec_name in vec_names:
    if b in vec_name:
        b_vecs.append(model[vec_names.index(vec_name)])

b_mean = np.mean(b_vecs, axis=0)

#substracting mean of paragraphs to a paragraph
print(model.docvecs.most_similar(positive = [model[vec_names.index(b + '_par_30')]], negative = [b_mean],topn=10))
'''