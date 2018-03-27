import glob
import os

print("Searching for books.")
book_filenames = sorted(glob.glob('../data/Subset/*/*txt'))

for book_filename in book_filenames:
    if len(open(book_filename, 'r').read()) < 50000:
        os.remove(book_filename)