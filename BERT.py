!pip install transformers

from transformers import BertTokenizer
# TOKENIZER

bert_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(bert_name, add_special_tokens=True, 
                                          do_lower_case=False, max_length=150, 
                                          pad_to_max_length=True)

tst = "This was an absolutely terrible movie. Don't be lured in \
        by Christopher Walken or Michael Ironside."



# ENCODE
tokens = tokenizer.encode(tst, add_special_tokens=True)
print(tokens)

#DECODE
print(tokenizer.decode(tokens))

" ".join([tokenizer.decode([tok]) for tok in tokens])

# MORE EXAMPLES

tokenizer.encode_plus(tst, add_special_tokens=True, max_length=150, 
                      pad_to_max_length=True, 
                      return_attention_mask=True, 
                      return_token_type_ids=True,
                      truncation=True)


tokenizer.encode_plus("Don't be lured", add_special_tokens=True, 
                      max_length=9,
                      pad_to_max_length=True, 
                      return_attention_mask=True, 
                      return_token_type_ids=True,
                      truncation=True)

tokenizer.encode_plus("Don't be"," lured", add_special_tokens=True, 
                      max_length=10,
                      pad_to_max_length=True, 
                      return_attention_mask=True, 
                      return_token_type_ids=True,
                      truncation=True
                     )

# LOADING DATA
import tensorflow_datasets as tfds
imdb_train, ds_info = tfds.load(name="imdb_reviews", split="train", 
                                with_info=True, as_supervised=True)
imdb_test = tfds.load(name="imdb_reviews", split="test", 
                      as_supervised=True)

imdb_train = imdb_train.take(100) 
imdb_test = imdb_test.take(100)

def bert_encoder(review):
    txt = review.numpy().decode('utf-8')
    encoded = tokenizer.encode_plus(txt, add_special_tokens=True, 
                                    max_length=150, pad_to_max_length=True, 
                                    return_attention_mask=True, 
                                    return_token_type_ids=True,
                                    truncation=True)
    return encoded['input_ids'], encoded['token_type_ids'], \
           encoded['attention_mask']


tst = imdb_train.take(5)
for review, label in tst:
    print(review , '\n', label)

bert_train = [bert_encoder(r) for r,l in imdb_train]
bert_lbl = [l for r, l in imdb_train]

print(bert_train[0])
bert_train = np.array(bert_train)
print(bert_train.shape)

bert_lbl = tf.keras.utils.to_categorical(bert_lbl, num_classes=2)
print(bert_lbl[0])
print(bert_lbl.shape)

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(bert_train, bert_lbl, 
                                                    test_size=0.2, 
                                                    random_state=42)

print(x_train.shape, y_train.shape)

tr_reviews, tr_segments, tr_masks = np.split(x_train, 3, axis=1)
val_reviews, val_segments, val_masks = np.split(x_val, 3, axis=1)
print(tr_reviews.shape)

tr_reviews = tr_reviews.squeeze()
tr_segments = tr_segments.squeeze()
tr_masks = tr_masks.squeeze()

val_reviews = val_reviews.squeeze()
val_segments = val_segments.squeeze()
val_masks = val_masks.squeeze()

def example_to_features(input_ids,attention_masks,token_type_ids,y):
  return {"input_ids": input_ids,
          "attention_mask": attention_masks,
          "token_type_ids": token_type_ids},y


train_ds = tf.data.Dataset.from_tensor_slices((tr_reviews, tr_masks, 
                                               tr_segments, y_train)).\
            map(example_to_features).shuffle(100).batch(16)

print(train_ds)
print(type(train_ds))


valid_ds = tf.data.Dataset.from_tensor_slices((val_reviews, val_masks, 
                                               val_segments, y_val)).\
            map(example_to_features).shuffle(100).batch(16)




