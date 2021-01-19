from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import json

df = pd.read_json("Sarcasm_Headlines_Dataset_v2.json", lines=True)


sentences = []
labels  = []
urls = []

for i in range (0,len(df)):
	sentences.append(df['headline'][i])
	labels.append(df['is_sarcastic'][i])
	urls.append(df['article_link'][i])

print(type(sentences))
print(labels)


tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)
print(word_index)