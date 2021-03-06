import numpy as np
import glob
from gensim.models.doc2vec import Doc2Vec
from scipy import spatial

vec_names = np.load('../models/vec_names_20k.npy').tolist()
book_filenames = sorted(glob.glob('../data/BookCorpus/*/*txt'))
# model = Doc2Vec.load('../models/par2vec_300_20k.doc2vec')
# par_vecs = model.docvecs.vectors_docs
target_vecs = np.load('../models/norm_par_vecs.npy')

#tree = spatial.KDTree(par_vecs)
tree = spatial.KDTree(target_vecs)


#cleaned_vecs = np.load('../models/book_par_vecs_20k.npy')
predicted_vecs = np.load('../models/b2p2v_predicted_vecs.npy')
#predicted_vecs_lstm = np.load('../models/b2p2v_predicted_vecs_lstm.npy')

#b = '../data/BookCorpus/Thriller/James_Bond-1.txt'
#b = '../data/BookCorpus/Thriller/Da_Vinci_Code.txt'
#b = '../data/BookCorpus/Romance/Attachments.txt'
#b = '../data/BookCorpus/Science_fiction/Asimov47.txt'
#b = '../data/BookCorpus/Fantasy/The_Hobbit.txt'
#b = '../data/BookCorpus/Fantasy/Stormlight_Archive-1.txt'
b = '../data/BookCorpus/Fantasy/Mistborn-1.txt'

#p = 0

for p in range(30):

    distances, closest = tree.query(predicted_vecs[book_filenames.index(b)][p], k= 2)

    #a = np.atleast_2d(par_vecs)
    #print("%d, %d" % (np.min(a), np.max(a)))

    print('Correct answer: book {} paragraph {}'.format((b.split('/')[4])[0:-4], str(p+2)))

    i = 0
    for c in closest:
        print(vec_names[c])
        print(distances[i])
        i += 1

    print('----------------------------')
    # print('LSTM:')
    #
    # distances, closest = tree.query(predicted_vecs_lstm[book_filenames.index(b)][p], k= 2)
    #
    # i = 0
    # for c in closest:
    #     print(vec_names[c])
    #     print(distances[i])
    #     i += 1
    #
    # print('____________________________')