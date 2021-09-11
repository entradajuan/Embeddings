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



