import numpy as np
from gensim.models.doc2vec import Doc2Vec
import glob

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

book_sequence_of_vectors = []

i = 0
for b in range(len(book_filenames)):
    bookseq = []
    for p in range(num_vec[b]):
        bookseq.append(model.docvecs[i])
        i += 1
    for q in range(num_vec[b], max(num_vec)):
        bookseq.append([0]*300)
    book_sequence_of_vectors.append(bookseq)

file_path = '../models/book_par_vecs_'+str(int(par_length/1000))+'k.npy'
print('Saving {} vectors of {} (for each paragraph of {}k words) for {} books under '.format(max(num_vec), vec_size, int(par_length/1000), len(book_filenames)) + file_path)

np.save(file_path, book_sequence_of_vectors)
np.save('../models/book_filenames.npy', book_filenames)
np.save('../models/num_vec.npy', num_vec)
