import numpy as np
import scipy.io
import random

#paragraph_corpus = np.load('../data/paragraph_corpus/Fantasy/The_Hobbit.npy')
#p1 = np.load('../data/paragraph_corpus/Science_fiction/Stormlight_Archive-1.npy')
#p2 = np.load('../data/paragraph_corpus/Science_fiction/u3019.npy')
p1 = np.load('../data/paragraph_corpus/Fantasy/Hunger_Games_trilogy_1.npy')


# list1 = np.array(p1[622 - 1], dtype=np.object)
# scipy.io.savemat('../matlab/compare_pars/stormlight622.mat', mdict={'par1':list1})
#
# list2 = np.array(p2[234 - 1], dtype=np.object)
# scipy.io.savemat('../matlab/compare_pars/goodOmens234.mat', mdict={'par2':list2})

#ind = [93,100,101,126,129,131,139,141,142,194,195,196,197,199,200,202]
#ind = [15,16,17,18,25,29,30,31,63,67,68,166,168,188]
# ind = random.sample(list(np.arange(200)), 10)
# random.shuffle(ind)
# pars=[]

print(' '.join(p1[190-1]))
print('')
#     pars.append(p1[i])
#     u = set.intersection(*map(set,pars))
#     print(u)
