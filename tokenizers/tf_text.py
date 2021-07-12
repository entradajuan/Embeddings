#!pip install tensorflow_text 

import tensorflow as tf
import tensorflow_text as tf_text

text_input = ["Madrid is the capital of Spain.",
              "I love to eat ice cream in Summer!!!",
              "What's your name?"]

def basic_preprocess(text_input, labels):
  tokenizer = tf_text.WhitespaceTokenizer()
  tokenized  = tokenizer.tokenize(text_input)
  
  return tokenized

basic_preprocess(text_input, 0)
