import glob
from gensim.models.doc2vec import Doc2Vec

book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))

model = Doc2Vec.load('../models/book2vec_full_300.doc2vec')

print(model.docvecs.most_similar(book_filenames.index('../data/BookCorpusFull/Thriller/Echo_Burning.txt')))

'''
for horror_book in sorted(glob.glob('../data/BookCorpus/Horror/*txt')):
    print(horror_book)
    print(model.docvecs.most_similar(book_filenames.index(horror_book)))
    print('\n \n \n')
'''

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