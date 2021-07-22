import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd

tf.__version__



tf.keras.backend.clear_session() #- for easy reset of notebook state

# chck if GPU can be seen by TF
tf.config.list_physical_devices('GPU')
# only if you want to see how commands are executed
#tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

#--------------------------------------------------------------------
# LOADING DATA

imdb_train, ds_info = tfds.load(name="imdb_reviews", split="train", 
                                with_info=True, as_supervised=True)
imdb_test = tfds.load(name="imdb_reviews", split="test", 
                      as_supervised=True)

print(type(imdb_train))
for example, label in imdb_train.take(1):
    print(example, '\n', label)


# TOKENIZER
#!pip install tensorflow_text 
#import tensorflow_text as tf_text

#vocabulary_set = set()
#MAX_TOKENS = 0

#for example, label in imdb_train.take(100):
#  tokenizer = tf_text.WhitespaceTokenizer()
#  some_tokens  = tokenizer.tokenize(example.numpy())
#
#  if MAX_TOKENS < len(some_tokens):
#        MAX_TOKENS = len(some_tokens)
#  vocabulary_set.update(some_tokens.numpy())

tokenizer = tfds.deprecated.text.Tokenizer()

vocabulary_set = set()
MAX_TOKENS = 0

for example, label in imdb_train:
  some_tokens = tokenizer.tokenize(example.numpy())
  if MAX_TOKENS < len(some_tokens):
        MAX_TOKENS = len(some_tokens)
  vocabulary_set.update(some_tokens)

print(MAX_TOKENS)
print(vocabulary_set)
print(len(vocabulary_set))

# ENCODER
imdb_encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set,
                                                    lowercase=True,
                                                    tokenizer=tokenizer)
vocab_size = imdb_encoder.vocab_size

print(vocab_size, MAX_TOKENS)

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

subset = imdb_train.take(10)
print(type(subset))
tst = subset.map(encode_tf_fn)
print(type(tst.take(10)))

for review, label in tst.take(1):
    print('>> ', review, label)
    print('>> ', imdb_encoder.decode(review))

encoded_train = imdb_train.map(encode_tf_fn,
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
encoded_test = imdb_test.map(encode_tf_fn,
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)

