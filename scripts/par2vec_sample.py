import numpy as np
from gensim.models.doc2vec import Doc2Vec

vec_names = np.load('../models/vec_names_50k.npy').tolist()

model = Doc2Vec.load('../models/par2vec_300_50k.doc2vec')

print(model.docvecs.most_similar(vec_names.index("../data/Subset/Fantasy/Mistborn-1_par_7")))