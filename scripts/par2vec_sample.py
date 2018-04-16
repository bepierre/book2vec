import numpy as np
from gensim.models.doc2vec import Doc2Vec
import glob


vec_names = np.load('../models/vec_names_full_1k.npy').tolist()
book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))

model = Doc2Vec.load('../models/par2vec_full_300_1k.doc2vec')

#b = '../data/BookCorpusFull/Fantasy/The_Hobbit'
b = '../data/BookCorpusFull/Mystery/Da_Vinci_Code'
#b = '../data/BookCorpus/Horror/Bullet'

#print(model[vec_names.index(b + '_par_110')])

print(model.docvecs.most_similar(positive = [model[vec_names.index(b + '_par_1000')]]))

'''
b_vecs = []
for vec_name in vec_names:
    if b in vec_name:
        b_vecs.append(model[vec_names.index(vec_name)])

b_mean = np.mean(b_vecs, axis=0)

#substracting mean of paragraphs to a paragraph
print(model.docvecs.most_similar(positive = [model[vec_names.index(b + '_par_30')]], negative = [b_mean],topn=10))
'''