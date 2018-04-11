import glob
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec
import matplotlib.pyplot as plt
import numpy as np

vec_size = 300
par_length = 1000

book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))
vec_names = np.load('../models/vec_names_full_'+str(int(par_length/1000))+'k.npy').tolist()

model = Doc2Vec.load('../models/par2vec_full_300_1k.doc2vec')

#b = '../data/BookCorpusFull/Thriller/James_Bond-1.txt'
#b = '../data/BookCorpusFull/Mystery/Da_Vinci_Code.txt'
#b = '../data/BookCorpusFull/Romance/Attachments.txt'
#b = '../data/BookCorpusFull/Science_fiction/Asimov47.txt'
b = '../data/BookCorpusFull/Fantasy/The_Hobbit.txt'
#b = '../data/BookCorpusFull/Science_fiction/Stormlight_Archive-1.txt'
#b = '../data/BookCorpusFull/Fantasy/Mistborn-1.txt'

start = vec_names.index(b[0:-4]+'_par_1')

next_book = book_filenames[book_filenames.index(b) + 1]

end = vec_names.index(next_book[0:-4]+'_par_1') - 1

par_vecs = model.docvecs.vectors_docs[start:end]

# b_mean = np.mean(model.docvecs.vectors_docs, axis=0)
# b_var = np.var(model.docvecs.vectors_docs, axis=0)
#
# norm_vecs = []
#
# for par_vec in par_vecs:
#     norm_vecs.append((par_vec - b_mean) / b_var)

centered_vecs = par_vecs - np.mean(par_vecs, axis=0)


#--------------------------- TSNE 2d

tsne_model_2D = TSNE(n_components=2, n_iter=10000, perplexity=300, init='pca')
tsne_articles_2D = tsne_model_2D.fit_transform(centered_vecs)

fig = plt.figure(figsize=(15,8))

order_scaled = np.arange(end-start)/(end-start)

colors = plt.cm.coolwarm(order_scaled)

plt.scatter([x[0] for x in tsne_articles_2D], [x[1] for x in tsne_articles_2D], c=colors)

for i in range(end-start-1):
    plt.plot([tsne_articles_2D[i][0], tsne_articles_2D[i+1][0]], [tsne_articles_2D[i][1], tsne_articles_2D[i+1][1]], '-', c=colors[i])

plt.plot(tsne_articles_2D[0][0], tsne_articles_2D[0][1], 'kx')

plt.legend()

plt.savefig('../figures/book_trajectories/'+(b.split('/')[4])[0:-4]+'_trajectory_centered_par_'+str(int(par_length/1000))+'k.png')

#--------------------------- collected trajectories

heatmap = np.matrix(np.array(par_vecs))
plt.imshow(heatmap, interpolation='nearest')
plt.savefig('../figures/book_trajectories/'+(b.split('/')[4])[0:-4]+'_heatmap_par_'+str(int(par_length/1000))+'k.pdf')

heatmap = np.matrix(np.array(centered_vecs))
plt.imshow(heatmap, interpolation='nearest')
plt.savefig('../figures/book_trajectories/'+(b.split('/')[4])[0:-4]+'_heatmap_centered_par_'+str(int(par_length/1000))+'k.pdf')
