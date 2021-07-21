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

!pip install tensorflow_text 
import tensorflow_text as tf_text

vocabulary_set = set()
MAX_TOKENS = 0

for example, label in imdb_train.take(100):
  tokenizer = tf_text.WhitespaceTokenizer()
  some_tokens  = tokenizer.tokenize(example.numpy())

  if MAX_TOKENS < len(some_tokens):
        MAX_TOKENS = len(some_tokens)
  vocabulary_set.update(some_tokens.numpy())


print(MAX_TOKENS)
print(vocabulary_set)
print(len(vocabulary_set))




