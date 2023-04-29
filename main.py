import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# isolate text column of dataset
d = pd.read_csv("depression.csv")
text = d["clean_text"]

# determine stopwords
stop_words = set(stopwords.words('english'))

# define function to remove stopwords
def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    filtered_text = " ".join(filtered_tokens)
    return filtered_text

text = text.apply(remove_stopwords)

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_text = " ".join(lemmatized_tokens)
    return lemmatized_text

text = text.apply(lemmatize_text)

stemmer = PorterStemmer()

def stem_text(text):
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    stemmed_text = " ".join(stemmed_tokens)
    return stemmed_text

text = text.apply(stem_text)

# np.savetxt("clean-dataset.csv", pd.Series.tolist(text))
print(text)