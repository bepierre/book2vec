import glob
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec
import matplotlib.pyplot as plt
import numpy as np

vec_size = 300
#par_length = 1000
words_per_par = 400

book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))
vec_names = np.load('../models/vec_names_full_'+str(int(words_per_par/100))+'c_w.npy').tolist()

#model = Doc2Vec.load('../models/par2vec'+size+'_'+str(vec_size)+'_'+str(int(par_length/1000))+'k.doc2vec')
model = Doc2Vec.load('../models/par2vec_full_300_4c_w.doc2vec')

#b = '../data/BookCorpusFull/Thriller/James_Bond-1.txt'
#b = '../data/BookCorpusFull/Mystery/Da_Vinci_Code.txt'
#b = '../data/BookCorpusFull/Romance/Marriage_Trap.txt'
#b = '../data/BookCorpusFull/Science_fiction/Asimov47.txt'
#b = '../data/BookCorpusFull/Fantasy/The_Hobbit.txt'
#b = '../data/BookCorpusFull/Science_fiction/Stormlight_Archive-1.txt'
#b = '../data/BookCorpusFull/Fantasy/Mistborn-1.txt'
#b = '../data/BookCorpusFull/Vampires/Vampireville.txt'
#b = '../data/BookCorpusFull/Fantasy/The_Silmarillon.txt'
b = '../data/BookCorpusFull/Vampires/Vampalicious.txt'

start = vec_names.index(b[0:-4]+'_par_1')

next_book = book_filenames[book_filenames.index(b) + 1]

end = vec_names.index(next_book[0:-4]+'_par_1') - 1

par_vecs = model.docvecs.vectors_docs[start:end]

np.savetxt("../models/book_trajectories/Vampalicious_4c_w.csv", par_vecs, delimiter=",")

# b_mean = np.mean(model.docvecs.vectors_docs, axis=0)
# b_var = np.var(model.docvecs.vectors_docs, axis=0)
#
# norm_vecs = []
#
# for par_vec in par_vecs:
#     norm_vecs.append((par_vec - b_mean) / b_var)

#centered_vecs = par_vecs - np.mean(par_vecs, axis=0)

# mat = np.matrix(np.array(centered_vecs))
#
# U, s, V = np.linalg.svd(mat)
#
# plt.plot(s)
#
# for i in range(5):
#     rand_mat = np.random.rand(mat.shape[0], mat.shape[1])
#     U, s, V = np.linalg.svd(rand_mat)
#     plt.plot(s)
#     print(s)
#
# plt.show()

#np.save('../models/norm_par_vecs'+size+'.npy', norm_vecs)