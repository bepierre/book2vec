import numpy as np
from gensim.models.doc2vec import Doc2Vec
import glob

size = '_full'
vec_size = 300
#par_length = 1000
words_per_par = 400

book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))
vec_names = np.load('../models/vec_names'+size +'_'+ str(int(words_per_par / 100)) + 'c_w.npy').tolist()
#genre_names = sorted(glob.glob('../data/BookCorpusFull/*'))

# for i in range(len(genre_names)):
#     genre_names[i] = genre_names[i].split('/')[3]
#
# book_genres = [0] * len(vec_names)
# j = 0
# for vec_name in vec_names:
#     print(vec_name)
#     book_genres[j] = genre_names.index(vec_name.split('/')[3])
#     j += 1

model = Doc2Vec.load('../models/par2vec'+size+'_'+str(vec_size)+'_'+str(int(words_per_par/100))+'c_w.doc2vec')

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


np.savetxt("../matlab/kmeans/num_vec.csv", num_vec, delimiter=",")
