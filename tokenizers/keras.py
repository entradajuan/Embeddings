import tensorflow as tf

sentences = ["Madrid is the capital of Spain.",
              "I love to eat ice cream in Summer!!!",
              "What's your name?"]

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

print(sequences)

print("Count of characters:",tokenizer.word_counts)
print("Length of text:",tokenizer.document_count)
print("Character index",tokenizer.word_index)
print("Frequency of characters:",tokenizer.word_docs)

sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences)

print(sequences)


