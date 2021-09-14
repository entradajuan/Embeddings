import tensorflow as tf

with open('kant.txt') as f:
    sentences = f.readlines()

#sentences = ["Madrid is the capital of Spain.",
#              "I love to eat ice cream in Summer!!!",
#              "What's your name?"]

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

print("Count of characters:",tokenizer.word_counts)
print("Length of text:",tokenizer.document_count)
print("Character index",tokenizer.word_index)
print("Frequency of characters:",tokenizer.word_docs)

sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

print(sequences)
print(type(sequences))
print(sequences.shape)

print("\n\n\n\n\n\n")
print(sequences[1])
print(tokenizer.sequences_to_texts([sequences[1]]))


