import argparse
import gensim.downloader as api
import numpy as np
import os
import shutil
import tensorflow as tf

from sklearn.metrics import accuracy_score, confusion_matrix

def download_and_read(url):
    local_file = url.split('/')[-1]
    p = tf.keras.utils.get_file(local_file, url, 
        extract=True, cache_dir=".")
    labels, texts = [], []
    local_file = os.path.join("datasets", "SMSSpamCollection")
    with open(local_file, "r") as fin:
        for line in fin:
            label, text = line.strip().split('\t')
            labels.append(1 if label == "spam" else 0)
            texts.append(text)
    return texts, labels


DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
texts, labels = download_and_read(DATASET_URL)

#TOKENIZE
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
tokenized_texts = tokenizer.texts_to_sequences(texts)
print(texts[0])
print(tokenized_texts[0])
print(tokenizer.sequences_to_texts([tokenized_texts[0]]))
text_sequences = tf.keras.preprocessing.sequence.pad_sequences(tokenized_texts)
num_records = len(texts)
max_seqlen = len(tokenized_texts[0])
NUM_CLASSES = 2
labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)

words_2_idx = tokenizer.word_index
words_2_idx
ids_2_words = {v:k for k, v in words_2_idx.items()}
print(words_2_idx['house'])
print(ids_2_words[words_2_idx['house']])

# preparar dataset
dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
dataset = dataset.shuffle(10000)
test_size = num_records // 4
val_size = (num_records - test_size) // 10
test_dataset = dataset.take(test_size)
val_dataset = dataset.skip(test_size).take(val_size)
train_dataset = dataset.skip(test_size + val_size)

BATCH_SIZE = 168
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

# embedding
EMBEDDING_MODEL = "glove-wiki-gigaword-300"

def build_embedding_matrix(sequences, word2idx, embedding_dim, embedding_file):
    if os.path.exists(embedding_file):
        E = np.load(embedding_file)
    else:
        vocab_size = len(word2idx)
        E = np.zeros((vocab_size, embedding_dim))
        word_vectors = api.load(EMBEDDING_MODEL)
        for word, idx in word2idx.items():
            try:
                E[idx] = word_vectors.word_vec(word)
            except KeyError:   # word not in embedding
                pass
            # except IndexError: # UNKs are mapped to seq over VOCAB_SIZE as well as 1
            #     pass
        np.save(embedding_file, E)
    return E


EMBEDDING_DIM = 300
DATA_DIR = "data"
EMBEDDING_NUMPY_FILE = os.path.join(DATA_DIR, "E.npy")
E = build_embedding_matrix(texts, words_2_idx, EMBEDDING_DIM,
    EMBEDDING_NUMPY_FILE)
print("Embedding matrix:", E.shape)




















