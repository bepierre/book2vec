import numpy as np

#paragraph_corpus = np.load('../data/paragraph_corpus/Fantasy/The_Hobbit.npy')
paragraph_corpus = np.load('../data/paragraph_corpus/Science_fiction/Stormlight_Archive-1.npy')

print(' '.join(paragraph_corpus[27]))
print(' '.join(paragraph_corpus[620]))