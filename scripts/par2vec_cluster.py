import numpy as np
from gensim.models.doc2vec import Doc2Vec
import glob
from sklearn.cluster import MiniBatchKMeans, KMeans

vec_names = np.load('../models/vec_names_full_4c_w.npy').tolist()
book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))

model = Doc2Vec.load('../models/par2vec_full_300_4c_w.doc2vec')

par_vecs = model.docvecs.vectors_docs

# book_means = [] # features
# start = 0
# curr_seq_length = -1
# for vec_name in vec_names:
#     if (int(vec_name.split('_')[-1]) == 1 and curr_seq_length > 0):
#         #print('averaging from {} to {}'.format(start, start+curr_seq_length))
#         #print(len(par_vecs[start:start+curr_seq_length]))
#         book_means.append(np.mean(par_vecs[start:start+curr_seq_length], axis=0))
#         start = start + curr_seq_length
#         curr_seq_length = 0
#     if int(vec_name.split('_')[-1]) > curr_seq_length:
#         curr_seq_length = int(vec_name.split('_')[-1])
# book_means.append(np.mean(par_vecs[start:start+curr_seq_length], axis=0))
#
# print('Computed book means to act as initial cluster centers of kmean. Starting kmeans clustering.')
#
# book_means = np.asarray(book_means)
#
# kmeans = MiniBatchKMeans(n_clusters=len(book_filenames), batch_size=100000, random_state=0, init=book_means).fit(par_vecs)
#
# np.savetxt("../models/kmeans_labels.csv", kmeans.labels_, delimiter=",")
# np.savetxt("../models/kmeans_cluster_centers.csv", kmeans.cluster_centers_, delimiter=",")
#
centered_vecs = []

start = 0
curr_seq_length = -1
for vec_name in vec_names:
    if (int(vec_name.split('_')[-1]) == 1 and curr_seq_length > 0):
        b_mean = np.mean(par_vecs[start:start+curr_seq_length], axis=0)
        for i in range(start, start+curr_seq_length):
            centered_vecs.append(par_vecs[i] - b_mean)
        start = start + curr_seq_length
        curr_seq_length = 0
    if int(vec_name.split('_')[-1]) > curr_seq_length:
        curr_seq_length = int(vec_name.split('_')[-1])
b_mean = np.mean(par_vecs[start:start+curr_seq_length], axis=0)
for i in range(start, start+curr_seq_length):
    centered_vecs.append(par_vecs[i] - b_mean)

print(len(centered_vecs))
print('Finished computing book centered paragraphs. Starting kmeans clustering.')

#kmeans_c = MiniBatchKMeans(n_clusters=100, batch_size=100000, random_state=0).fit(centered_vecs)
kmeans_c = KMeans(n_clusters=100, random_state=0).fit(centered_vecs)

np.save("../models/kmeans_100_c_labels.npy", kmeans_c.labels_)
np.save("../models/kmeans_100_c_centers.npy", kmeans_c.cluster_centers_)

# np.savetxt("../models/kmeans_10_c_labels.csv", kmeans_c.labels_, delimiter=",")
# np.savetxt("../models/kmeans_10_c_cluster_centers.csv", kmeans_c.cluster_centers_, delimiter=",")
#
# diff_vecs = []
#
# start = 0
# curr_seq_length = -1
# for vec_name in vec_names:
#     if (int(vec_name.split('_')[-1]) == 1 and curr_seq_length > 0):
#         diff_vecs.extend(par_vecs[start:start+curr_seq_length-1] - par_vecs[start+1:start+curr_seq_length])
#         start = start + curr_seq_length
#         curr_seq_length = 0
#     if int(vec_name.split('_')[-1]) > curr_seq_length:
#         curr_seq_length = int(vec_name.split('_')[-1])
# diff_vecs.extend(par_vecs[start:start+curr_seq_length-1] - par_vecs[start+1:start+curr_seq_length])
#
# print(len(diff_vecs))
# print('Finished computing book difference between paragraphs. Starting kmeans clustering.')
#
# kmeans_d = MiniBatchKMeans(n_clusters=len(book_filenames), batch_size=100000, random_state=0).fit(diff_vecs)
#
# np.savetxt("../models/kmeans_d_labels.csv", kmeans_d.labels_, delimiter=",")
# np.savetxt("../models/kmeans_d_cluster_centers.csv", kmeans_d.cluster_centers_, delimiter=",")