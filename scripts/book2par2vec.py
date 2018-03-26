import glob
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing

print("Searching for books.")
book_filenames = sorted(glob.glob('../data/Subset/*/*txt'))

'''
paragraph_corpus = []
errors = 0
for book_filename in book_filenames:
    with open(book_filename, 'r') as book_file:
        paragraph_corpus.append(
            TaggedDocument(
                gensim.utils.simple_preprocess(
                    book_file.read()), ["{}".format(book_filename)]))
'''