import argparse
import gensim.downloader as api
import numpy as np
import os
import shutil
import tensorflow as tf

from sklearn.metrics import accuracy_score, confusion_matrix

##_______________________________________________________________________


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


##_______________________________________________________________________


DATA_DIR = "data"
EMBEDDING_NUMPY_FILE = os.path.join(DATA_DIR, "E.npy")
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
EMBEDDING_MODEL = "glove-wiki-gigaword-300"
EMBEDDING_DIM = 300
NUM_CLASSES = 2
BATCH_SIZE = 128
NUM_EPOCHS = 3

# data distribution is 4827 ham and 747 spam (total 5574), which 
# works out to approx 87% ham and 13% spam, so we take reciprocals
# and this works out to being each spam (1) item as being approximately
# 8 times as important as each ham (0) message.
CLASS_WEIGHTS = { 0: 1, 1: 8 }

tf.random.set_seed(42)

#parser = argparse.ArgumentParser()
#parser.add_argument("--mode", help="run mode",
#    choices=[
#        "scratch",
#        "vectorizer",
#        "finetuning"
#    ])
#args = parser.parse_args()
#run_mode = args.mode

run_mode = "finetuning"

texts, labels = download_and_read(DATASET_URL)

print(type(texts))
print(texts[0])
print(labels[0])

# tokenize and pad text
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
text_sequences = tokenizer.texts_to_sequences(texts)

print(text_sequences[0])

text_sequences = tf.keras.preprocessing.sequence.pad_sequences(text_sequences)

print(text_sequences[0])
print(text_sequences[1])
print(text_sequences[2])

num_records = len(text_sequences)
max_seqlen = len(text_sequences[0])
print("{:d} sentences, max length: {:d}".format(num_records, max_seqlen))

# labels
cat_labels = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)

print(cat_labels)

# vocabulary
word2idx = tokenizer.word_index
print(word2idx)
idx2word = {v:k for k, v in word2idx.items()}
print(idx2word)
word2idx["PAD"] = 0
idx2word[0] = "PAD"
vocab_size = len(word2idx)
print("vocab size: {:d}".format(vocab_size))

# dataset
dataset = tf.data.Dataset.from_tensor_slices((text_sequences, cat_labels))
print(dataset)
print(type(dataset))


dataset = dataset.shuffle(10000)
test_size = num_records // 4
val_size = (num_records - test_size) // 10
test_dataset = dataset.take(test_size)
val_dataset = dataset.skip(test_size).take(val_size)
train_dataset = dataset.skip(test_size + val_size)

test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

