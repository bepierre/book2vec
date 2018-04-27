import glob
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np
import scipy.io

# def chunks(l, n):
#     """Yield successive n-sized chunks from l."""
#     for i in range(0, len(l), n):
#         yield l[i:i + n]
#
# size = '_full'
# vec_size = 300
# words_per_par = 400
#
# size= '_full'
#
# paragraph_corpus = []
# vec_names = []
# book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))
# vec_names = np.load('../models/vec_names'+size +'_'+ str(int(words_per_par / 100)) + 'c_w.npy').tolist()
# genre_names = sorted(glob.glob('../data/BookCorpusFull/*'))
#
# books = []
# for book_filename in book_filenames:
#     books.append( book_filename.split('/')[4][0:-4].replace('_', ' '))
#
# list2 = np.array(books, dtype=np.object)
# scipy.io.savemat('../matlab/kmeans/book_names.mat', mdict={'book_names':list2})

# book_filename = '../data/BookCorpusFull/Mystery/Da_Vinci_Code.txt'
# with open(book_filename, 'r') as book_file:
#     book = gensim.utils.simple_preprocess(book_file.read())
#     pars = list(chunks(book, words_per_par))
#     #remove first paragraph because it contains book info
#     #remove last paragraph because it might be too short
#     pars = pars[1:-1]
#     p=1
#     for par in pars:
#         paragraph_corpus.append(
#                            TaggedDocument(
#                                  par, [book_filename[0:len(book_filename)-4] + '_par_' + str(p)]))
#         vec_names.append(book_filename[0:len(book_filename)-4] + '_par_' + str(p))
#         # if (book_filename[0:len(book_filename) - 4] + '_par_' + str(p)=='../data/BookCorpusFull/Thriller/James_Bond-1_par_32'):
#         #     print(par)
#         #     exit(0)
#         p += 1
#
# print(paragraph_corpus[0])

labels = np.loadtxt("../models/kmeans_100_c_labels.csv", delimiter=",")
np.save("../models/kmeans_100_c_labels.npy", labels)
