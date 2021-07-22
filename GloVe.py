!wget http://nlp.stanford.edu/data/glove.6B.zip

!unzip glove.6B.zip

dict_w2v = {}
with open('glove.6B.50d.txt', "r") as file:
    for line in file:
        tokens = line.split()
        word = tokens[0]
        vector = np.array(tokens[1:], dtype=np.float32)

        if vector.shape[0] == 50:
            dict_w2v[word] = vector
        else:
            print("There was an issue with " + word)

# lets check the vocabulary size
print("Dictionary Size: ", len(dict_w2v))


