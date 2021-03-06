import glob
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

book_vecs = np.load('../models/bpc_book_vecs.npy')
book_filenames = np.load('../models/eval_book_filenames_full.npy')
genre_names = sorted(glob.glob('../data/BookCorpusFull/*'))

for i in range(len(genre_names)):
    genre_names[i] = genre_names[i].split('/')[3]


book_genres = [0] * len(book_filenames)
j = 0
for book_filename in book_filenames:
    book_genres[j] = genre_names.index(book_filename.split('/')[3])
    j += 1

vec_size = 100

tsne_model_2D = TSNE(n_components=2, n_iter=10000, perplexity=22, init='pca')
tsne_articles_2D = tsne_model_2D.fit_transform(book_vecs)

# 2D
fig = plt.figure(figsize=(15,8))

for g in range(len(genre_names)):
    indexes = list(range(book_genres.index(g),book_genres.index(g)+book_genres.count(g)))
    plt.scatter([x[0] for x in tsne_articles_2D[indexes,]], [x[1] for x in tsne_articles_2D[indexes,]], label = genre_names[g])

plt.legend()


plt.savefig('../figures/cluster_2D_bpc_trained_on_full_'+str(vec_size)+'.png')