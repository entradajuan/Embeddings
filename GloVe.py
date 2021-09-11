import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd

tf.__version__

!wget http://nlp.stanford.edu/data/glove.6B.zip

!unzip glove.6B.zip

dict_w2v = {}
with open('glove.6B.50d.txt', "r") as file:
    for line in file:
        tokens = line.split()
        word = tokens[0]
        vector = np.array(tokens[1:], dtype=np.float32)

        if vector.shape[0] == 50:
            dict_w2v[word] = vector
        else:
            print("There was an issue with " + word)

len(dict_w2v['house'])
dict_w2v['bank']
embedding_dim = len(dict_w2v['house'])


# LOADING DATA

imdb_train, ds_info = tfds.load(name="imdb_reviews", split="train", 
                                with_info=True, as_supervised=True)
imdb_test = tfds.load(name="imdb_reviews", split="test", 
                      as_supervised=True)


print(type(imdb_train))
for sentence, label in imdb_train.take(1):
    print(sentence, '\n', label)

tokenizer = tfds.deprecated.text.Tokenizer()


vocabulary_set = set()
MAX_TOKENS = 0

for sentence, label in imdb_train:
  some_tokens = tokenizer.tokenize(sentence.numpy())
  if MAX_TOKENS < len(some_tokens):
        MAX_TOKENS = len(some_tokens)
  vocabulary_set.update(some_tokens)


print(len(vocabulary_set))
# ENCODER
imdb_encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set,
                                                    lowercase=True,
                                                    tokenizer=tokenizer)

embedding_matrix = np.zeros((imdb_encoder.vocab_size, embedding_dim))
print(embedding_matrix)

unk_cnt = 0
unk_set = set()
for word in imdb_encoder.tokens:
    embedding_vector = dict_w2v.get(word)

    if embedding_vector is not None:
        tkn_id = imdb_encoder.encode(word)[0]
        embedding_matrix[tkn_id] = embedding_vector
    else:
        unk_cnt += 1
        unk_set.add(word)

word = 'house'
print(imdb_encoder.encode(word))
print(imdb_encoder.encode(word)[0])
print(dict_w2v.get(word))

print("Total unknown words: ", unk_cnt)


from tensorflow.keras.preprocessing import sequence

def encode_pad_transform(sample):
    encoded = imdb_encoder.encode(sample.numpy())
    pad = sequence.pad_sequences([encoded], padding='post', 
                                 maxlen=150)
    return np.array(pad[0], dtype=np.int64)  



def encode_tf_fn(sentence, label):
    encoded = tf.py_function(encode_pad_transform, 
                                       inp=[sentence], 
                                       Tout=(tf.int64))
    encoded.set_shape([None])
    label.set_shape([])
    return encoded, label


encoded_train = imdb_train.map(encode_tf_fn,
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
encoded_test = imdb_test.map(encode_tf_fn,
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)



for e in imdb_train.take(1):
  print(e)

for e in encoded_train.take(1):
  print(e)

print([e for e in imdb_train.take(1)][0][0].numpy().decode(encoding = 'utf-8').split()[0])
print([e for e in encoded_train.take(1)][0][0].numpy()[0])
print(embedding_matrix[[e for e in encoded_train.take(1)][0][0].numpy()[0]])

