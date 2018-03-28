import glob
from gensim.models.doc2vec import Doc2Vec


book_filenames = sorted(glob.glob('../data/BookCorpus/*/*txt'))

model = Doc2Vec.load('../models/book2vec_10.doc2vec')

print(model.docvecs.most_similar(book_filenames.index("../data/BookCorpus/Fantasy/Mistborn-1.txt")))

'''
script_filenames = sorted(glob.glob('../data/ScriptCorpus/*/*txt'))

model = Doc2Vec.load('../models/script2vec_300.doc2vec')

print(model.docvecs.most_similar(script_filenames.index("../data/ScriptCorpus/Fantasy/batman.txt")))
'''

'''
for book_filename in book_filenames:
    most_similar = model.docvecs.most_similar(book_filenames.index(book_filename))[0:1][0]
    print("{} - {}".format(book_filename, most_similar))
'''