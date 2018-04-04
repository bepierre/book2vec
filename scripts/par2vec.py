import glob
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing
import numpy as np

print("Searching for books.")
book_filenames = sorted(glob.glob('../data/BookCorpus/*/*.txt'))

par_length = 20000

paragraph_corpus = []
vec_names = []
for book_filename in book_filenames:
    with open(book_filename, 'r') as book_file:
        pars = list(map(''.join, zip(*[iter(book_file.read())] * par_length)))
        p = 1
        for par in pars:
            paragraph_corpus.append(
                TaggedDocument(
                    gensim.utils.simple_preprocess(
                        par), [book_filename[0:len(book_filename)-4] + '_par_' + str(p)]))
            vec_names.append(book_filename[0:len(book_filename)-4] + '_par_' + str(p))
            p += 1

np.save('../models/vec_names_'+str(int(par_length/1000))+'k.npy', vec_names)

print(str(len(book_filenames)) + ' Books split into ' + str(len(paragraph_corpus)) + ' paragraphs of length: ' + str(int(par_length/1000)) + 'k.')

words = 0
for paragraph in paragraph_corpus:
    words += len(paragraph[0])

print('Total number of words: ' + str(words))

cores = multiprocessing.cpu_count()

vec_size = 300

model = Doc2Vec(size = vec_size, min_count = 5, workers=cores, alpha = 0.025, min_alpha=0.025, iter=5)
#model = Doc2Vec(size = 300, min_count = 5, workers=cores, iter = 10)


model.build_vocab(paragraph_corpus)

print('Vocabulary length: ' +  str(len(model.wv.vocab)))

print('Starting to train with ' + str(cores) + ' cpu cores.')

#model.train(book_corpus, total_examples=model.corpus_count, epochs=model.iter)

epochs = 40
for epoch in range(epochs):
    model.train(paragraph_corpus, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.0005
    model.min_alpha = model.alpha
    print('Finished epoch ' + str(epoch + 1) + ' out of ' + str(epochs))

    model_name =  '../models/par2vec_'+str(vec_size)+'_'+str(int(par_length/1000))+'k.doc2vec'
    model.save(model_name)
    print('Saved model under ' + model_name)