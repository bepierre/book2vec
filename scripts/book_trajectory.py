import glob
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec
import matplotlib.pyplot as plt
import numpy as np

vec_size = 300
par_length = 1000

book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))
vec_names = np.load('../models/vec_names_full_'+str(int(par_length/1000))+'k.npy').tolist()

model = Doc2Vec.load('../models/par2vec__full300_1k.doc2vec')
tsne_model_2D = TSNE(n_components=2, n_iter=10000, perplexity=100, init='pca')

#b = '../data/BookCorpusFull/Thriller/James_Bond-1.txt'
b = '../data/BookCorpusFull/Mystery/Da_Vinci_Code.txt'
#b = '../data/BookCorpusFull/Romance/Attachments.txt'
#b = '../data/BookCorpusFull/Science_fiction/Asimov47.txt'
#b = '../data/BookCorpusFull/Fantasy/The_Hobbit.txt'
#b = '../data/BookCorpusFull/Fantasy/Stormlight_Archive-1.txt'
#b = '../data/BookCorpusFull/Fantasy/Mistborn-1.txt'

start = vec_names.index(b[0:-4]+'_par_1')

next_book = book_filenames[book_filenames.index(b) + 1]

end = vec_names.index(next_book[0:-4]+'_par_1') - 1

tsne_articles_2D = tsne_model_2D.fit_transform(model.docvecs.vectors_docs[start:end])

fig = plt.figure(figsize=(15,8))

order_scaled = np.arange(end-start)/(end-start)

colors = plt.cm.coolwarm(order_scaled)

plt.scatter([x[0] for x in tsne_articles_2D], [x[1] for x in tsne_articles_2D], c=colors)

for i in range(end-start-1):
    plt.plot([tsne_articles_2D[i][0], tsne_articles_2D[i+1][0]], [tsne_articles_2D[i][1], tsne_articles_2D[i+1][1]], '-', c=colors[i])

plt.plot(tsne_articles_2D[0][0], tsne_articles_2D[0][1], 'kx')

plt.legend()

plt.savefig('../figures/book_trajectories/'+(b.split('/')[4])[0:-4]+'_trajectory_par_'+str(int(par_length/1000))+'k.png')
