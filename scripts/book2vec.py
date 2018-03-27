import glob
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing

print("Searching for books.")
book_filenames = sorted(glob.glob('../data/Subset/*/*txt'))

book_corpus = []
errors = 0
for book_filename in book_filenames:
    try:
        with open(book_filename, 'r') as book_file:
            book_corpus.append(
                TaggedDocument(
                    gensim.utils.simple_preprocess(
                        book_file.read()), ['{}'.format(book_filename)]))
    except:
        errors += 1


print('Corrupted books: ' + str(errors) + ' out of ' + str(len(book_filenames)))

words = 0
for book in book_corpus:
    words += len(book[0])

print('Total number of words: ' + str(words))

cores = multiprocessing.cpu_count()

vec_size = 50

model = Doc2Vec(size = vec_size, min_count = 5, workers=cores, alpha = 0.025, min_alpha=0.025, iter=10)
#model = Doc2Vec(size = 300, min_count = 5, workers=cores, iter = 10)


model.build_vocab(book_corpus)

print('Vocabulary length: ' +  str(len(model.wv.vocab)))

print('Starting to train with ' + str(cores) + ' cpu cores.')

#model.train(book_corpus, total_examples=model.corpus_count, epochs=model.iter)

epochs = 20
for epoch in range(epochs):
    model.train(book_corpus, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.001
    model.min_alpha = model.alpha
    print('Finished epoch ' + str(epoch + 1) + ' out of ' + str(epochs))

    model_name =  '../models/book2vec_'+str(vec_size)+'.doc2vec'
    model.save(model_name)
    print('Saved model under ' + model_name)