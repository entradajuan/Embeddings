!pip install tensorflow==1.15
!pip install "tensorflow_hub>=0.6.0"
!pip3 install tensorflow_text==1.15

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text

tf.compat.v1.disable_eager_execution()
elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

embeddings = elmo([
        "i like green eggs and ham",
        "would you eat them in a box"
    ], 
    signature="default",
    as_dict=True)

print(type(embeddings))
print(type(embeddings["elmo"]))
print(embeddings["elmo"])


print(embeddings["elmo"].shape)
