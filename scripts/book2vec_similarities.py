import glob
from gensim.models.doc2vec import Doc2Vec
import matplotlib.pyplot as plt


book_filenames = sorted(glob.glob('../data/BookCorpus/*/*txt'))

model = Doc2Vec.load('../models/book2vec_300.doc2vec')

#for each article, find the cosine similarity to each other article
cosine_similarities = []
for r in range(len(book_filenames)):
    for t in range(r+1, len(book_filenames)):
        cosine_similarities.append(model.docvecs.similarity(r,t))

plt.hist(cosine_similarities, 50, facecolor='green', alpha=0.5)
plt.title('Distribution of Cosine Similarities')
plt.ylabel('Frequency')
plt.xlabel('Cosine Similarity')
plt.savefig('../figures/similarities_hist.png')