import numpy as np
from gensim.models.doc2vec import Doc2Vec
import glob

book_filenames = sorted(glob.glob('../data/Subset/*/*txt'))
vec_names = np.load('../models/vec_names_20k.npy').tolist()
genre_names = sorted(glob.glob('../data/Subset/*'))

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

seq_lengths = [0]*len(book_filenames)
curr_seq_length = 0
b = -1
for vec_name in vec_names:
    if int(vec_name.split('_')[-1]) == 1:
        print(vec_name)
        b += 1
        print(book_filenames[b])
        curr_seq_length = 0
    if int(vec_name.split('_')[-1]) > curr_seq_length:
        curr_seq_length = int(vec_name.split('_')[-1])
        seq_lengths[b] = curr_seq_length

print(len(model.docvecs.vectors_docs))

book_sequence_of_vectors = [[[0] * 300]*max(seq_lengths)]*len(book_filenames)



