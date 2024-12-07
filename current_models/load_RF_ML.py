import pickle
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.corpus import stopwords as sw
import string
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV

lemmatizer = WordNetLemmatizer()

stopwords = sw.words('english')



# Load the fitted model
with open('model_RF_ML_binary/saved_model_new.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def clean_text(text):
    # First clean and tokenize the text
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    # Lemmatize and remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
    return " ".join(tokens)  # Return joined text instead of tokens

def count_punct(text):
    text_length = len(text) - text.count(" ")
    if text_length == 0:
        return 0
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / text_length, 3) * 100


# Load the fitted TfidfVectorizer
with open('model_RF_ML_binary/tfidf_vect_new.pkl', 'rb') as file:
    tfidf_vect = pickle.load(file)

def make_prediction(input_text):
    # Prepare the input
    input_data = pd.DataFrame([input_text], columns=['sentence'])
    input_data['body_len'] = input_data['sentence'].apply(lambda x: len(x) - x.count(" "))
    input_data['punct%'] = input_data['sentence'].apply(lambda x: count_punct(x))
    input_data['CAPS%'] = input_data['sentence'].apply(
        lambda x: len([x for x in x.split() if x.isupper()]) / len(x.split()) * 100 if len(x.split()) != 0 else 0)
    input_data['word_count'] = input_data['sentence'].apply(lambda x: len(x.split()))
    input_data['all_Caps'] = input_data['sentence'].apply(lambda x: len([x for x in x.split() if x.isupper()]))


    # Transform the input
    X_input_tfidf = tfidf_vect.transform(input_data['sentence'])
    X_input_tfidf_feat = pd.concat(
        [input_data['body_len'], input_data['punct%'], input_data['CAPS%'],input_data['word_count'],input_data['all_Caps'], pd.DataFrame(X_input_tfidf.toarray())],
        axis=1)

    # Make sure the columns are of type string
    X_input_tfidf_feat.columns = X_input_tfidf_feat.columns.astype(str)

    # Make the prediction
    prediction = loaded_model.predict_proba(X_input_tfidf_feat)

    return prediction

loop = True
while loop:
    input_text = input("Please enter your input text: ")
    if input_text == "exit":
        loop = False
        break
    prediction = make_prediction(input_text)
    print(prediction)
