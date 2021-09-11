import tensorflow as tf
import tensorflow_datasets as tfds

# LOADING DATA

imdb_train, ds_info = tfds.load(name="imdb_reviews", split="train", 
                                with_info=True, as_supervised=True)
imdb_test = tfds.load(name="imdb_reviews", split="test", 
                      as_supervised=True)

# TOKENIZE
tokenizer = tfds.deprecated.text.Tokenizer()
vocabulary_set = set()
MAX_TOKENS = 0
for sentence, label in imdb_train:
  some_tokens = tokenizer.tokenize(sentence.numpy())
  if MAX_TOKENS < len(some_tokens):
        MAX_TOKENS = len(some_tokens)
  vocabulary_set.update(some_tokens)

print('VOCAB SIZE = ', len(vocabulary_set))

df = tfds.as_dataframe(imdb_train, ds_info)
print(df.head())
print(df.describe())
print(df.shape)

label_train = df.iloc[:, :1]
features_train = df.iloc[:, 1:]

class CBOWModel(tf.keras.Model):
    def __init__(self, vocab_sz, emb_sz, window_sz, **kwargs):
        super(CBOWModel, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_sz,
            output_dim=emb_sz,
            embeddings_initializer="glorot_uniform",
            input_length=window_sz*2
        )
        self.dense = tf.keras.layers.Dense(
            vocab_sz,
            kernel_initializer="glorot_uniform",
            activation="softmax"
        )

    def call(self, x):
        x = self.embedding(x)
        x = tf.reduce_mean(x, axis=1)
        x = self.dense(x)
        return x


VOCAB_SIZE = len(vocabulary_set)
EMBED_SIZE = 300
WINDOW_SIZE = 1  # 3 word window, 1 on left, 1 on right

model = CBOWModel(VOCAB_SIZE, EMBED_SIZE, WINDOW_SIZE)
model.build(input_shape=(None, VOCAB_SIZE))
model.compile(optimizer=tf.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"])

model.summary()

# train the model here

##   OJITO!!!!!   ENTENDER LO QUE SE QUIERE HACER AQUI!!!
features_train = [w for w in vocabulary_set]
model.fit(features_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=True)



# retrieve embeddings from trained model
emb_layer = [layer for layer in model.layers 
    if layer.name.startswith("embedding")][0]
emb_weight = [weight.numpy() for weight in emb_layer.weights
    if weight.name.endswith("/embeddings:0")][0]
print(emb_weight, emb_weight.shape)






























