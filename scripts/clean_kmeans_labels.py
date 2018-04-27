import numpy as np
from gensim.models.doc2vec import Doc2Vec
import glob
import random

size = '_full'
vec_size = 300
#par_length = 1000
words_per_par = 400


book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))
vec_names = np.load('../models/vec_names'+size +'_'+ str(int(words_per_par / 100)) + 'c_w.npy').tolist()
labels = np.load("../models/kmeans_100_c_labels.npy")

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

book_sequence_of_labels = []


i = 0
for b in range(len(book_filenames)):
    bookseq = []
    for p in range(num_vec[b]):
        bookseq.append(int(labels[i]))
        i += 1
    for q in range(num_vec[b], max(num_vec)):
        bookseq.append(0)
    book_sequence_of_labels.append(bookseq)


random.seed(10)
eval_samples = random.sample(range(len(book_filenames)), 500)
#print(eval_samples)

num_vecs_per_book = max(num_vec)

eval_sequence_of_labels = [x for i, x in enumerate(book_sequence_of_labels) if i in eval_samples]
eval_book_filenames = [x for i, x in enumerate(book_filenames) if i in eval_samples]
eval_num_vec = [x for i, x in enumerate(num_vec) if i in eval_samples]

book_sequence_of_labels = [x for i, x in enumerate(book_sequence_of_labels) if i not in eval_samples]
book_filenames = [x for i, x in enumerate(book_filenames) if i not in eval_samples]
num_vec = [x for i, x in enumerate(num_vec) if i not in eval_samples]

file_path = '../models/book_labels'+ size +'_' + str(int(words_per_par / 100)) + 'c_w'+'.npy'
print('Saving {} labels (for each paragraph of {} hundred words) for {} books under '.format(num_vecs_per_book, int(words_per_par/100), len(book_filenames)) + file_path)
np.save(file_path, book_sequence_of_labels)
np.save('../models/book_filenames'+ size +'_' + str(int(words_per_par / 100)) + 'c_w.npy', book_filenames)
np.save('../models/num_vec'+ size +'_' + str(int(words_per_par / 100)) + 'c_w.npy', num_vec)

file_path = '../models/eval_labels'+ size +'_'  + str(int(words_per_par / 100)) + 'c_w.npy'
print('Saving {} labels (for each paragraph of {} hundred words) for {} books (evaluation set) under '.format(num_vecs_per_book, int(words_per_par/100), len(eval_book_filenames)) + file_path)
np.save(file_path, eval_sequence_of_labels)
np.save('../models/eval_book_filenames'+ size +'_' + str(int(words_per_par / 100)) + 'c_w.npy', eval_book_filenames)
np.save('../models/eval_num_vec'+ size +'_' + str(int(words_per_par / 100)) + 'c_w.npy', eval_num_vec)