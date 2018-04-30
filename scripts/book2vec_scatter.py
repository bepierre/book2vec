import glob
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec
import matplotlib.pyplot as plt
import numpy as np

#book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))
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

model = Doc2Vec.load('../models/book2vec_eval_'+str(vec_size)+'.doc2vec')
#tsne_model_2D = TSNE(n_components=2, random_state=0, verbose=1, init="pca", n_iter=10000, perplexity=50)
tsne_model_2D = TSNE(n_components=2, n_iter=10000, perplexity=50, init='pca')
tsne_articles_2D = tsne_model_2D.fit_transform(model.docvecs.vectors_docs)

# 2D
fig = plt.figure(figsize=(15,8))

#plt.scatter([x[0] for x in tsne_articles_2D], [x[1] for x in tsne_articles_2D], c = book_genres)

for g in range(len(genre_names)):
    indexes = list(range(book_genres.index(g),book_genres.index(g)+book_genres.count(g)))
    plt.scatter([x[0] for x in tsne_articles_2D[indexes,]], [x[1] for x in tsne_articles_2D[indexes,]], label = genre_names[g])

plt.legend()

plt.savefig('../figures/cluster_2D_eval_'+str(vec_size)+'.png')

# 3D
#fig2 = plt.figure(figsize=(15,8))

#ax = fig2.add_subplot(111, projection='3d')
#ax.scatter([x[0] for x in tsne_articles_3D], [x[1] for x in tsne_articles_3D], [x[2] for x in tsne_articles_3D], c = book_genres)

#plt.savefig('cluster_3D.png')
