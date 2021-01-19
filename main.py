import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
'I love my dog',
'I love my cat',
'You love my dog!',
'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>") # The num_words parameter is the maximum number of words to keep
# OOV will create a token for that, and then replace words that it does not recognize with the Out Of Vocabulary token instead.
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, padding='post', truncating='post', maxlen = 5) #padding='post' might want them after the sentence
# maxlen if you don't wat the length of the padded sentences to be the same as the longest sentence
# truncaing On supprime le début de la phrase lonque que '5'

# [8 6 9 2 4]

print(word_index)
print(sequences)
print(padded)
