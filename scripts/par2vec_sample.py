import numpy as np
from gensim.models.doc2vec import Doc2Vec

vec_names = np.load('../models/vec_names_20k.npy').tolist()

model = Doc2Vec.load('../models/par2vec_300_20k.doc2vec')

print(model.docvecs.most_similar(vec_names.index("../data/Subset/Fantasy/Ace_in_the_Hole_par_39")))