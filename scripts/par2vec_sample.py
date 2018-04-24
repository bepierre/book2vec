import numpy as np
from gensim.models.doc2vec import Doc2Vec
import glob


vec_names = np.load('../models/vec_names_full_4c_w.npy').tolist()
book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))

model = Doc2Vec.load('../models/par2vec_full_300_4c_w.doc2vec')

#b = '../data/BookCorpusFull/Thriller/James_Bond-1.txt'
#b = '../data/BookCorpusFull/Mystery/Da_Vinci_Code.txt'
#b = '../data/BookCorpusFull/Romance/Marriage_Trap.txt'
#b = '../data/BookCorpusFull/Science_fiction/Asimov47.txt'
#b = '../data/BookCorpusFull/Fantasy/The_Hobbit.txt'
b = '../data/BookCorpusFull/Science_fiction/Stormlight_Archive-1.txt'
#b = '../data/BookCorpusFull/Fantasy/Mistborn-1.txt'
#b = '../data/BookCorpusFull/Vampires/Vampireville.txt'

#print(model[vec_names.index(b + '_par_110')])


start = vec_names.index(b[0:-4]+'_par_1')
next_book = book_filenames[book_filenames.index(b) + 1]
end = vec_names.index(next_book[0:-4]+'_par_1') - 1
par_vecs = model.docvecs.vectors_docs[start:end]
mean = np.mean(par_vecs, axis=0)
# print(model.docvecs.most_similar(positive = [mean]))

#print(model.docvecs.most_similar(vec_names.index(b[:-4] + '_par_100')))

max_sim = 0
for i in range(start, end):
    most_similar = model.docvecs.most_similar(i)[0:1][0]
    max_sim = max(most_similar[1], max_sim)
    print("{} - {}".format(vec_names[i], most_similar))

print(max_sim)



# b_vecs = []
# for vec_name in vec_names:
#     if b in vec_name:
#         b_vecs.append(model[vec_names.index(vec_name)])
#
# b_mean = np.mean(b_vecs, axis=0)
#
# #substracting mean of paragraphs to a paragraph
# print(model.docvecs.most_similar(positive = [model[vec_names.index(b + '_par_30')]], negative = [b_mean],topn=10))
