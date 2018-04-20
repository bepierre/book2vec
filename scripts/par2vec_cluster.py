import numpy as np
from gensim.models.doc2vec import Doc2Vec
import glob
from sklearn.cluster import MiniBatchKMeans

vec_names = np.load('../models/vec_names_full_4c_w.npy').tolist()
book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))

model = Doc2Vec.load('../models/par2vec_full_300_4c_w.doc2vec')

par_vecs = model.docvecs.vectors_docs

book_means = [] # features *
start = 0
curr_seq_length = -1
for vec_name in vec_names:
    if (int(vec_name.split('_')[-1]) == 1 and curr_seq_length > 0):
        #print('averaging from {} to {}'.format(start, start+curr_seq_length-1))
        for i in range(curr_seq_length):
            book_means.append(np.mean(par_vecs[start:start+curr_seq_length-1,], axis=0))
        start = start + curr_seq_length
        curr_seq_length = 0
    if int(vec_name.split('_')[-1]) > curr_seq_length:
        curr_seq_length = int(vec_name.split('_')[-1])
for i in range(curr_seq_length):
    book_means.append(np.mean(par_vecs[start:start+curr_seq_length,], axis=0))

print('Computed book means to act as initial cluster centers of kmean.')
print('Kmean starts.')

kmeans = MiniBatchKMeans(n_clusters=len(book_filenames), batch_size=10000, random_state=0, init=book_means).fit(par_vecs)

np.savetxt("../models/kmeans_labels.csv", kmeans.labels_, delimiter=",")
np.savetxt("../models/kmeans_cluster_centers.csv", kmeans.cluster_centers_, delimiter=",")

# centered_vecs = []s
#
# for b in book_filenames:
#     b = b[0:-4]
#     b_vecs = []
#     for vec_name in vec_names:
#         if b in vec_name:
#             b_vecs.append(model[vec_names.index(vec_name)])
#
#     b_mean = np.mean(b_vecs, axis=0)
#     for vec_name in vec_names:
#         if b in vec_name:
#             #print(vec_name)
#             centered_vecs.append((model[vec_names.index(vec_name)] - b_mean))
#
# print('Finished computing book centered paragraphs. Starting kmeans clustering.')
#
# kmeans = MiniBatchKMeans(n_clusters=len(book_filenames), batch_size=100000, random_state=0).fit(centered_vecs)
#
# np.savetxt("../models/kmeans_c_labels.csv", kmeans.labels_, delimiter=",")
# np.savetxt("../models/kmeans_c_cluster_centers.csv", kmeans.cluster_centers_, delimiter=",")
