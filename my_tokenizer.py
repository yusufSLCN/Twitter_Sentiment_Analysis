from sentimentPredictor import preprocess
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

max_length = 300
vocab_size = 10000
embedding_dim = 32

DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
data = pd.read_csv("./training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", names=DATASET_COLUMNS)
print('read')
data.text = data.text.apply(lambda x: preprocess(x, stem=True))
df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)
print('read')
tokenizer = Tokenizer(num_words=vocab_size, oov_token="OOV")
tokenizer.fit_on_texts(df_train.text)

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


