import json
import nltk
import re
import string
import numpy as np
import pandas as pd
from flask import jsonify
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer


def avg_word(sentence):
    words = sentence.split()
    return sum(len(word) for word in words)/len(words)


def feature_extraction(data, column='title'):
    data['word_count'] = data[column].apply(lambda x: len(str(x).split(" ")))
    data['char_count'] = data[column].str.len()
    data['avg_word'] = data[column].apply(lambda x: avg_word(x))
    stop_words_collection = set(stopwords.words('english'))
    data['stopwords'] = data[column].apply(lambda x: len([x for x in x.split() if x in stop_words_collection]))
    data['numerics'] = data[column].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
    data['upper'] = data[column].apply(lambda x: len([x for x in x.split() if x.isupper()]))
    fex = data[[column, 'word_count', 'char_count', 'avg_word', 'stopwords', 'numerics', 'upper']]
    return fex


def cleaning_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def cleaning_data(data, column='title'):
    data[column] = data[column].apply(lambda x: cleaning_text(x))
    return data


def stop_words_removal(text, column='title'):
    stop = stopwords.words('english')
    text[column] = text[column].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    return text


def tokenization(text, column='title'):
    text['tokenization'] = text[column].apply(lambda x: TextBlob(x).words)
    return text[[column, 'tokenization']]


def stemming(text, column='title'):
    st = PorterStemmer()
    text['stemming'] = text[column][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    return text[[column, 'stemming']]


def lemmatization(text, column='title'):
    text['lemmatization'] = text[column].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return text[[column, 'lemmatization']]


def n_gram(text, column='title'):
    text['n-gram'] = text[column].apply(lambda x: TextBlob(x).ngrams(2))
    return text[[column, 'n-gram']]


def term_of_frequency_1(text, column='title'):
    tf1 = (text[column][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
    tf1.columns = ['words', 'tf']
    return tf1


def inverse_tf(text, column='title'):
    tf1 = term_of_frequency_1(text, column)
    for i, words in enumerate(tf1['words']):
        a = len(text[text[column].str.contains(words)])
        if a == 0:
            print("Error ZeroDivisionError")
        else:
            tf1.loc[i, 'idf'] = np.log(text.shape[0] / a)
    return tf1


def tf_idf(text, column):
    tf1 = inverse_tf(text, column)
    tf1['tfidf'] = tf1['tf'] * tf1['idf']
    return tf1


def bag_of_words(text, column='title'):
    bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1, 1), analyzer='word')
    bow.fit_transform(text[column])
    return text


def sentiment_analysis(text, column='title'):
    text[column][:5].apply(lambda x: TextBlob(x).sentiment)
    text['sentiment'] = text[column].apply(lambda x: TextBlob(x).sentiment[0])
    res = text[[column, 'sentiment']]
    return res


def sentiment_analysis_process(data, column='title'):

    feature_extraction_clt = feature_extraction(data, column).to_json(orient='records')
    jsonify(feature_extraction_clt)
    data_cleaning = cleaning_data(data, column)
    new_data = stop_words_removal(data_cleaning, column)
    tokenization_data = tokenization(new_data, column)
    stemming_data = stemming(new_data, column)
    lemmatization_data = lemmatization(new_data, column)
    n_gram_data = n_gram(new_data, column)
    tf_idf_data = tf_idf(new_data, column)
    bag_of_words_data = bag_of_words(new_data, column)
    text_pre_processing = {
            "data_cleaning": data_cleaning.to_json(),
            "stop_words_removal": new_data.to_json(),
            "tokenization": tokenization_data.to_json(),
            "stemming": stemming_data.to_json(),
            "lemmatization": lemmatization_data.to_json(),
            "n-grams": n_gram_data.to_json(),
            "tf_idf": tf_idf_data.to_json(),
            "bag_of_words": bag_of_words_data.to_json()
        }

    sentiment_analysis_data = sentiment_analysis(new_data).to_json(orient='records')
    type(sentiment_analysis_data)
    data_sa = {
            "feature_extraction": feature_extraction_clt,
            "text_pre_processing": text_pre_processing,
            "sentiment_analysis": sentiment_analysis_data
        }

    final_res = json.dumps(data_sa, sort_keys=True)

    return final_res
