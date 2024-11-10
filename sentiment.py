from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


import pickle
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.corpus import stopwords as sw
import string
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
import requests

lemmatizer = WordNetLemmatizer()

stopwords = sw.words('english')

# Load the fitted model
with open('./model_117k_est200_max90/saved_model_new.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
    return text


def count_punct(text):
    text_length = len(text) - text.count(" ")
    if text_length == 0:
        return 0
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / text_length, 3) * 100


# Load the fitted TfidfVectorizer
with open('./model_117k_est200_max90/tfidf_vect_new.pkl', 'rb') as file:
    tfidf_vect = pickle.load(file)


def make_prediction(input_text):
    # Prepare the input
    input_data = pd.DataFrame([input_text], columns=['sentence'])
    input_data['body_len'] = input_data['sentence'].apply(lambda x: len(x) - x.count(" "))
    input_data['punct%'] = input_data['sentence'].apply(lambda x: count_punct(x))
    input_data['CAPS%'] = input_data['sentence'].apply(
        lambda x: len([x for x in x.split() if x.isupper()]) / len(x.split()) * 100 if len(x.split()) != 0 else 0)

    # Transform the input
    X_input_tfidf = tfidf_vect.transform(input_data['sentence'])
    X_input_tfidf_feat = pd.concat(
        [input_data['body_len'], input_data['punct%'], input_data['CAPS%'], pd.DataFrame(X_input_tfidf.toarray())],
        axis=1)

    # Make sure the columns are of type string
    X_input_tfidf_feat.columns = X_input_tfidf_feat.columns.astype(str)

    # Make the prediction
    prediction = loaded_model.predict(X_input_tfidf_feat)

    return prediction


@app.route('/sentiment/<name>', methods=['GET'])
def classify_reviews(name):
    result = []
    response = requests.get(f'http://localhost:5000/reviews/{name}')
    if response.status_code == 200:
        reviews = response.json()
        for review in reviews[name]:
            for sentence in review['text'].split('.'):
                if sentence.strip():
                    sentiment = make_prediction(sentence)
                    sentiment = sentiment.tolist()
                    result.append({sentence: sentiment})
    else:
        return jsonify({"error": "Failed to fetch reviews"}), 500
    result = {'result': result}
    return jsonify(result)


@app.route('/sentiment', methods=['POST'])
def classify_review():
    data = request.get_json()
    input_text = data.get('text', '')

    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    sentences = input_text.split('.')
    sentences = [sentence for sentence in sentences if sentence.strip()]
    results = []
    scores = []

    for sentence in sentences:
        if sentence.strip():
            sentiment = make_prediction(sentence)
            sentiment = sentiment.tolist()
            pos, neg = sentiment[0]
            
            sentiment_label = "positive" if pos > neg*-1 else "negative" if neg*-1 > pos else "neutral"
            results.append({
                "sentence": sentence,
                "pos": pos,
                "neg": neg,
                "sentiment": sentiment_label
            })
            scores.append({"pos": pos, "neg": neg})

    response = {
        "scores": scores,
        "results": results
    }
    print(response)

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, threaded=True)  # Runs the Flask server accessible from other devices