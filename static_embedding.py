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

model = KeyedVectors.load('data/text8-word2vec.bin')
word_vecs = model.wv

print(type(word_vecs))
words = word_vecs.vocab.keys()

w_list = [w for w in words ] 
#w_list = [w for i,w in enumerate(words) if i > 10] 
print(w_list)
print(len(w_list))

similars_to_king = word_vecs.most_similar('King'.lower())
for i, (word, rate) in enumerate(similars_to_king):
  print(word, ' ', rate)
  if (i>=5):
    break
if (len(similars_to_king) > 5):
  print('...')

res1 = word_vecs.most_similar(positive=['paris', 'madrid'], negative=['france'])
print(res1)

res1 = word_vecs.most_similar_cosmul(positive=['france', 'madrid'], negative=['paris'])
print(res1)

res1 = word_vecs.most_similar_cosmul(positive=['king', 'woman'], negative=['man'])
print(res1)

res1 = word_vecs.doesnt_match(['madrid', 'sevilla', 'barcelona', 'computer' ])
print(res1)

res1 = word_vecs.doesnt_match(['madrid', 'sevilla', 'barcelona', 'seattle' ])
print(res1)

for w in ["woman", "dog", "whale", "tree"]:
  print('man sim', w, 'is',  word_vecs.similarity("man", w))
#  print('man cos sim', w, 'is',  word_vecs.cosine_similarities(word_vecs["man"], word_vecs[w]))


print("# distance between vectors")
print("distance(singapore, malaysia) = {:.3f}".format(
    word_vecs.distance("singapore", "malaysia")
))

vec_song = word_vecs["song"]
print("\n# output vector obtained directly, shape:", vec_song.shape)

vec_song_2 = word_vecs.word_vec("song", use_norm=True)
print("# output vector obtained using word_vec, shape:", vec_song_2.shape)

