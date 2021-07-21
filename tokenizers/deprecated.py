import tensorflow as tf
import tensorflow_datasets as tfds

text_input = ["Madrid is the capital of Spain.",
              "I love to eat ice cream in Summer!!!",
              "What's your name?"]

# TOKENIZER
tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
MAX_TOKENS = 0

for example in text_input:
  some_tokens = tokenizer.tokenize(example.numpy())
  if MAX_TOKENS < len(some_tokens):
        MAX_TOKENS = len(some_tokens)
  vocabulary_set.update(some_tokens)



# ENCODER
imdb_encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set,
                                                   lowercase=True,
                                                   tokenizer=tokenizer)
vocab_size = imdb_encoder.vocab_size

print(vocab_size, MAX_TOKENS)


