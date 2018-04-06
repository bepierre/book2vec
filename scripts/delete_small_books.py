import glob
import os

print("Searching for books.")
book_filenames = sorted(glob.glob('../data/BookCorpusFull/*/*txt'))

for book_filename in book_filenames:
    try:
        if len(open(book_filename, 'r').read()) < 50000:
            print('too short')
            os.remove(book_filename)
    except:
        print('corrupted')
        os.remove(book_filename)