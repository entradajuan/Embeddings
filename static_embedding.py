import gensim.downloader as api
from gensim.models import Word2Vec

info = api.info("text8")
print(info)
assert(len(info) > 0)

dataset = api.load("text8")
print(dataset)
model = Word2Vec(dataset)

import os
output_dir = "data/"

if not os.path.exists(output_dir):
  os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

model.save("data/text8-word2vec.bin")

from gensim.models import KeyedVectors

model =KeyedVectors.load('data/text8-word2vec.bin')
word_vecs = model.wv

print(type(word_vecs))
words = word_vecs.vocab.keys()

w_list = [w for w in words ] 
#w_list = [w for i,w in enumerate(words) if i > 10] 
print(w_list)
print(len(w_list)

