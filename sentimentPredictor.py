import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re


DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

max_length = 300
vocab_size = 10000
embedding_dim = 32
stop_words = stopwords.words('english')
stemmer = SnowballStemmer("english")


def preprocess(text, stem=False):
    # Remove link,user and special characters
    # text = re.sub(TEXT_CLEANING_RE, " ", text)
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return ' '.join(tokens)


if __name__ == '__main__':
    data = pd.read_csv("./training.1600000.processed.noemoticon.csv", encoding="ISO-8859-1", names=DATASET_COLUMNS)
    data.text = data.text.apply(lambda x: preprocess(x, stem=True))
    data.target = data.target/4
    print('processed')

    #divide test and training set
    df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="OOV")

    tokenizer.fit_on_texts(df_train.text)
    train_seq = tokenizer.texts_to_sequences(df_train.text)
    train_padded = pad_sequences(train_seq, maxlen=max_length, padding='post', truncating='post')
    train_label = df_train.target

    test_seq = tokenizer.texts_to_sequences(df_test.text)
    test_padded = pad_sequences(test_seq, maxlen=max_length, padding='post', truncating='post')
    test_label = df_test.target

    #prepare the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size,embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'Recall'])
    model.summary()
    print('ready to fit')
    num_epochs = 15
    history = model.fit(train_padded,train_label, batch_size=64, epochs=num_epochs,validation_data=(test_padded,test_label),verbose=2)
    model.save('./models/sentPred_embedDim32_vocab100000_2')
