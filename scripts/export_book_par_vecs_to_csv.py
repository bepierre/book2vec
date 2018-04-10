import glob
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec
import matplotlib.pyplot as plt
import numpy as np

vec_size = 300
par_length = 1000

book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))
vec_names = np.load('../models/vec_names_full_'+str(int(par_length/1000))+'k.npy').tolist()

#model = Doc2Vec.load('../models/par2vec'+size+'_'+str(vec_size)+'_'+str(int(par_length/1000))+'k.doc2vec')
model = Doc2Vec.load('../models/par2vec__full300_1k.doc2vec')

#b = '../data/BookCorpusFull/Thriller/James_Bond-1.txt'
#b = '../data/BookCorpusFull/Mystery/Da_Vinci_Code.txt'
#b = '../data/BookCorpusFull/Romance/Attachments.txt'
#b = '../data/BookCorpusFull/Science_fiction/Asimov47.txt'
b = '../data/BookCorpusFull/Fantasy/The_Hobbit.txt'
#b = '../data/BookCorpusFull/Science_fiction/Stormlight_Archive-1.txt'
#b = '../data/BookCorpusFull/Fantasy/Mistborn-1.txt'

start = vec_names.index(b[0:-4]+'_par_1')

next_book = book_filenames[book_filenames.index(b) + 1]

end = vec_names.index(next_book[0:-4]+'_par_1') - 1

par_vecs = model.docvecs.vectors_docs[start:end]

b_mean = np.mean(model.docvecs.vectors_docs, axis=0)
b_var = np.var(model.docvecs.vectors_docs, axis=0)

norm_vecs = []

for par_vec in par_vecs:
    norm_vecs.append((par_vec - b_mean) / b_var)

#np.save('../models/norm_par_vecs'+size+'.npy', norm_vecs)

print(par_vecs[0])

#np.savetxt("../models/book_trajectories/Hobbit_1k.csv", par_vecs, delimiter=",")
