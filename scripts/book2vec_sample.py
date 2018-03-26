import glob
from gensim.models.doc2vec import Doc2Vec

book_filenames = sorted(glob.glob('../data/Subset/*/*txt'))

model = Doc2Vec.load('../models/mymodel_300.doc2vec')

#print(model.docvecs.most_similar(book_filenames.index("Documents/BookCorpus/Subset/Romance/Wild_About_You_1.txt")))
print(model.docvecs.most_similar(book_filenames.index("../data/Subset/Fantasy/Mistborn-1.txt")))



'''
for book_filename in book_filenames:
    most_similar = model.docvecs.most_similar(book_filenames.index(book_filename))[0:1][0]
    print("{} - {}".format(book_filename, most_similar))
'''