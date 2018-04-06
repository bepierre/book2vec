import numpy as np
from gensim.models.doc2vec import Doc2Vec
import glob
import random

book_filenames = sorted(glob.glob('../data/BookCorpus/*/*txt'))
vec_names = np.load('../models/vec_names_20k.npy').tolist()
genre_names = sorted(glob.glob('../data/BookCorpus/*'))

for i in range(len(genre_names)):
    genre_names[i] = genre_names[i].split('/')[3]

book_genres = [0] * len(vec_names)
j = 0
for vec_name in vec_names:
    book_genres[j] = genre_names.index(vec_name.split('/')[3])
    j += 1


vec_size = 300
par_length = 20000

model = Doc2Vec.load('../models/par2vec_'+str(vec_size)+'_'+str(int(par_length/1000))+'k.doc2vec')

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

np.save('../models/norm_par_vecs.npy', norm_vecs)

book_sequence_of_vectors = []

i = 0
for b in range(len(book_filenames)):
    bookseq = []
    for p in range(num_vec[b]):
        bookseq.append(norm_vecs[i])
        i += 1
    for q in range(num_vec[b], max(num_vec)):
        bookseq.append([0]*300)
    book_sequence_of_vectors.append(bookseq)

eval_samples = random.sample(range(len(book_filenames)), 30)

eval_sequence_of_vectors = [x for i, x in enumerate(book_sequence_of_vectors) if i in eval_samples]
eval_book_filenames = [x for i, x in enumerate(book_filenames) if i in eval_samples]
eval_num_vec = [x for i, x in enumerate(num_vec) if i in eval_samples]

book_sequence_of_vectors = [x for i, x in enumerate(book_sequence_of_vectors) if i not in eval_samples]
book_filenames = [x for i, x in enumerate(book_filenames) if i not in eval_samples]
num_vec = [x for i, x in enumerate(num_vec) if i not in eval_samples]

file_path = '../models/book_norm_par_vecs_' + str(int(par_length / 1000)) + 'k.npy'
print('Saving {} vectors of {} (for each paragraph of {}k words) for {} books under '.format(max(num_vec), vec_size, int(par_length/1000), len(book_filenames)) + file_path)

np.save(file_path, book_sequence_of_vectors)
np.save('../models/book_filenames.npy', book_filenames)
np.save('../models/num_vec.npy', num_vec)

file_path = '../models/eval_norm_par_vecs_' + str(int(par_length / 1000)) + 'k.npy'
print('Saving {} vectors of {} (for each paragraph of {}k words) for {} books (evaluation set) under '.format(max(num_vec), vec_size, int(par_length/1000), len(eval_book_filenames)) + file_path)

np.save(file_path, eval_sequence_of_vectors)
np.save('../models/eval_book_filenames.npy', eval_book_filenames)
np.save('../models/eval_num_vec.npy', eval_num_vec)