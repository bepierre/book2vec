# import numpy as np
# from gensim.models.doc2vec import Doc2Vec
# import glob
# import random
#
# size = ''
# vec_size = 300
# #par_length = 1000
# words_per_par = 400
#
#
# book_filenames = sorted(glob.glob('../data/BookCorpus/*/*txt'))
# vec_names = np.load('../models/vec_names'+size +'_'+ str(int(words_per_par / 100)) + 'c_w.npy').tolist()
# labels = np.load("../models/kmeans_100_c_labels.npy")
#
# b_full = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))
# b = sorted(glob.glob('../data/BookCorpus/*/*txt'))
# subset_samples = [b_full.index(i) for i in book_filenames]
# print(subset_samples)
# print(len(subset_samples))
# exit(0)
#
# labels = [x for i, x in enumerate(labels) if i in subset_samples]
#
# num_vec = [0] * len(book_filenames)
# curr_seq_length = 0
# b = -1
# for vec_name in vec_names:
#     if int(vec_name.split('_')[-1]) == 1:
#         b += 1
#         curr_seq_length = 0
#     if int(vec_name.split('_')[-1]) > curr_seq_length:
#         curr_seq_length = int(vec_name.split('_')[-1])
#         num_vec[b] = curr_seq_length
#
# book_sequence_of_labels = []
#
#
# i = 0
# for b in range(len(book_filenames)):
#     bookseq = []
#     for p in range(num_vec[b]):
#         bookseq.append(int(labels[i]))
#         i += 1
#     for q in range(num_vec[b], max(num_vec)):
#         bookseq.append(0)
#     book_sequence_of_labels.append(bookseq)
#
# num_vecs_per_book = max(num_vec)
#
# file_path = '../models/subset_labels'+ size +'_' + str(int(words_per_par / 100)) + 'c_w'+'.npy'
# print('Saving {} labels (for each paragraph of {} hundred words) for {} books under '.format(num_vecs_per_book, int(words_per_par/100), len(book_filenames)) + file_path)
# np.save(file_path, book_sequence_of_labels)
# np.save('../models/subset_filenames'+ size +'_' + str(int(words_per_par / 100)) + 'c_w.npy', book_filenames)
# np.save('../models/subset_num_vec'+ size +'_' + str(int(words_per_par / 100)) + 'c_w.npy', num_vec)
#
