import glob
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import os

print("Searching for books.")
book_filenames = sorted(glob.glob('../data/Subset/*/*txt'))
book_filepaths = sorted(glob.glob('../data/Subset/*/*txt'))

for i in range(len(book_filenames)):
    book_filenames[i] = book_filenames[i].split('/')[4]

for book_filename1 in book_filenames:
    for book_filename2 in book_filenames[book_filenames.index(book_filename1)+1:]:
        if book_filename1 == book_filename2:
            print(book_filepaths[book_filenames.index(book_filename1)])
