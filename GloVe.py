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

# lets check the vocabulary size
print("Dictionary Size: ", len(dict_w2v))

print(dict_w2v['cat'])
print(len(dict_w2v['cat']))

print(dict_w2v['dog'])
print(len(dict_w2v['dog']))

embedding_dim = 50
embedding_matrix = np.zeros((imdb_encoder.vocab_size, embedding_dim))

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

# Print how many werent found
print("Total unknown words: ", unk_cnt)

import tensorflow_datasets as tfds

# LOADING DATA

imdb_train, ds_info = tfds.load(name="imdb_reviews", split="train", 
                                with_info=True, as_supervised=True)
imdb_test = tfds.load(name="imdb_reviews", split="test", 
                      as_supervised=True)

print(type(imdb_train))
for example, label in imdb_train.take(1):
    print(example, '\n', label)


# TOKENIZER

tokenizer = tfds.deprecated.text.Tokenizer()

vocabulary_set = set()
MAX_TOKENS = 0

for example, label in imdb_train:
  some_tokens = tokenizer.tokenize(example.numpy())
  if MAX_TOKENS < len(some_tokens):
        MAX_TOKENS = len(some_tokens)
  vocabulary_set.update(some_tokens)

# ENCODER
imdb_encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set,
                                                    lowercase=True,
                                                    tokenizer=tokenizer)

vocab_size = imdb_encoder.vocab_size # 



from tensorflow.keras.preprocessing import sequence

def encode_pad_transform(sample):
    encoded = imdb_encoder.encode(sample.numpy())
    pad = sequence.pad_sequences([encoded], padding='post', 
                                 maxlen=150)
    return np.array(pad[0], dtype=np.int64)  


def encode_tf_fn(sample, label):
    encoded = tf.py_function(encode_pad_transform, 
                                       inp=[sample], 
                                       Tout=(tf.int64))
    encoded.set_shape([None])
    label.set_shape([])
    return encoded, label

    
encoded_train = imdb_train.map(encode_tf_fn,
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
encoded_test = imdb_test.map(encode_tf_fn,
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)




# Number of RNN units
rnn_units = 64

#batch size
BATCH_SIZE=100



from tensorflow.keras.layers import Embedding, LSTM, \
                                    Bidirectional, Dense,\
                                    Dropout
            
def build_model_bilstm(vocab_size, embedding_dim, 
                       rnn_units, batch_size, train_emb=False):
  model = tf.keras.Sequential([
    Embedding(vocab_size, embedding_dim, mask_zero=True,
              weights=[embedding_matrix], trainable=train_emb),
    #Dropout(0.25),  
    Bidirectional(tf.keras.layers.LSTM(rnn_units, return_sequences=True, 
                                      dropout=0.5)),
    Bidirectional(tf.keras.layers.LSTM(rnn_units, dropout=0.25)),
    Dense(1, activation='sigmoid')
  ])
  return model


model_fe = build_model_bilstm(
  vocab_size = vocab_size,
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

model_fe.summary()

model_fe.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy', 'Precision', 'Recall'])

encoded_train_batched = encoded_train.batch(BATCH_SIZE).prefetch(100)


model_fe.fit(encoded_train_batched, epochs=30)

print(model_fe.evaluate(encoded_test.batch(BATCH_SIZE)))


model_ft = build_model_bilstm(
  vocab_size=vocab_size,
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE,
  train_emb=True)

model_ft.summary()

model_ft.compile(loss='binary_crossentropy', 
             optimizer='adam', 
             metrics=['accuracy', 'Precision', 'Recall'])

model_ft.fit(encoded_train_batched, epochs=50)

print(model_ft.evaluate(encoded_test.batch(BATCH_SIZE)))















