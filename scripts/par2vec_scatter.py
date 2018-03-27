import glob
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec
import matplotlib.pyplot as plt
import numpy as np

book_filenames = sorted(glob.glob('../data/Subset/*/*txt'))
vec_names = np.load('../models/vec_names_50k.npy').tolist()
genre_names = sorted(glob.glob('../data/Subset/*'))

for i in range(len(genre_names)):
    genre_names[i] = genre_names[i].split('/')[3]

book_genres = [0] * len(vec_names)
j = 0
for vec_name in vec_names:
    book_genres[j] = genre_names.index(vec_name.split('/')[3])
    j += 1


load = False

vec_size = 300
par_length = 50000

if not load:
    model = Doc2Vec.load('../models/par2vec_'+str(vec_size)+'_'+str(int(par_length/1000))+'k.doc2vec')
    tsne_model_2D = TSNE(n_components=2, n_iter=10000, perplexity=100)
    tsne_articles_2D = tsne_model_2D.fit_transform(model.docvecs.vectors_docs)
    np.save('../models/tsne2D_'+str(vec_size)+'_par.npy', tsne_articles_2D)
else:
    tsne_articles_2D = np.load('../models/tsne2D_'+str(vec_size)+'_par.npy')

# 2D
fig = plt.figure(figsize=(15,8))

for g in range(len(genre_names)):
    indexes = list(range(book_genres.index(g),book_genres.index(g)+book_genres.count(g)))
    plt.scatter([x[0] for x in tsne_articles_2D[indexes,]], [x[1] for x in tsne_articles_2D[indexes,]], label = genre_names[g])

plt.legend()

plt.savefig('../figures/cluster_2D_'+str(vec_size)+'_par.png')
