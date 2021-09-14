import numpy as np
import pandas as pd

data = ["Madrid is the capital of Spain.",
              "I love to eat ice cream in Summer!!!",
              "What's your name?"]


corpus = np.asarray(data)
print(corpus)
print(corpus.shape)

vocab = set()
for sentence in corpus:
  words_list = sentence.split()
  for word in words_list:
    vocab.add(word)

print("Vocab size = ", len(vocab))
word_2_idx = {word:ind for ind, word in enumerate(vocab)}
idx_2_word = {ind:word for ind, word in enumerate(vocab)}

print(idx_2_word[10])

# Aunque para hacer mapping mejor un Dataframe!!!!
#______________________________________________________________________________
corpus = pd.DataFrame(data)


split_sentence = lambda sentence : sentence.split()

def my_encoder(words_list):
  tokenized = []
  for w in words_list:
    tokenized.append(word_2_idx[w])
  return tokenized

corpus[0] = corpus[0].map(split_sentence).map(my_encoder) 

MAX_LENGTH = corpus[0].map(len).describe().max()

print(corpus.head())

def padding(words_list):
  dif = int(MAX_LENGTH) - len(words_list)
  for i in range(dif):
    words_list.append(0)
  return words_list

corpus[0] = [padding(words_list) for words_list in corpus[0]]


print(corpus)
