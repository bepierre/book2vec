import numpy as np
from gensim.models.doc2vec import Doc2Vec
import glob
import random

size = '_full'

book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))
vec_names = np.load('../models/vec_names'+size+'_20k.npy').tolist()
genre_names = sorted(glob.glob('../data/BookCorpusFull/*'))

for i in range(len(genre_names)):
    genre_names[i] = genre_names[i].split('/')[3]

book_genres = [0] * len(vec_names)
j = 0
for vec_name in vec_names:
    book_genres[j] = genre_names.index(vec_name.split('/')[3])
    j += 1


vec_size = 300
par_length = 20000

model = Doc2Vec.load('../models/par2vec'+size+'_'+str(vec_size)+'_'+str(int(par_length/1000))+'k.doc2vec')

#print(list(model.docvecs.doctags.keys())[0])
#print(vec_names[0])

num_vec = [0] * len(book_filenames)
curr_seq_length = 0
b = -1
for vec_name in vec_names:
    if int(vec_name.split('_')[-1]) == 1:
        b += 1
        curr_seq_length = 0
    if int(vec_name.split('_')[-1]) > curr_seq_length:
        curr_seq_length = int(vec_name.split('_')[-1])
        num_vec[b] = curr_seq_length

par_vecs = model.docvecs.vectors_docs

norm_vecs = []

b_mean = np.mean(par_vecs, axis=0)
b_var = np.var(par_vecs, axis=0)

for vec_name in vec_names:
    norm_vecs.append((model[vec_names.index(vec_name)] - b_mean) / b_var)

np.save('../models/norm_par_vecs'+size+'.npy', norm_vecs)

book_sequence_of_vectors = []

i = 0
for b in range(len(book_filenames)):
    bookseq = []
    for p in range(num_vec[b]):
        bookseq.append(norm_vecs[i])
        i += 1
    book_sequence_of_vectors.append(bookseq)

np.savetxt("../models/Hobbit.csv", book_sequence_of_vectors[book_filenames.index('../data/BookCorpusFull/Fantasy/The_Hobbit.txt')], delimiter=",")
np.savetxt("../models/DaVinciCode.csv", book_sequence_of_vectors[book_filenames.index('../data/BookCorpusFull/Mystery/Da_Vinci_Code.txt')], delimiter=",")