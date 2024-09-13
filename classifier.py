import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.corpus import stopwords as sw
import string
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV

lemmatizer = WordNetLemmatizer()


myDataset = pd.read_json("results(4).json")
#print(myDataset)

# Remove rows where 'sentence' is an empty string
myDataset = myDataset[myDataset['sentence'] != '']

# Reset the index
myDataset = myDataset.reset_index(drop=True)

print(myDataset)

stopwords = sw.words('english')

def count_punct(text):
    text_length = len(text) - text.count(" ")
    if text_length == 0:
        return 0
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / text_length, 3) * 100

myDataset['body_len'] = myDataset['sentence'].apply(lambda x: len(x) - x.count(" "))
myDataset['punct%'] = myDataset['sentence'].apply(lambda x: count_punct(x))
myDataset['CAPS%'] = myDataset['sentence'].apply(lambda x: len([x for x in x.split() if x.isupper()]) / len(x.split()) * 100 if len(x.split()) != 0 else 0)
#pd.set_option('display.max_columns', None)
myDataset['score'] = myDataset.apply(lambda row: 'positive' if row['pos'] > row['neg']*-1 else 'negative', axis=1)
print(myDataset.head(10))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
    return text

# myDataset = myDataset[0:18000]

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
X_tfidf = tfidf_vect.fit_transform(myDataset['sentence'])
X_tfidf_feat = pd.concat([myDataset['body_len'], myDataset['punct%'], myDataset['CAPS%'], pd.DataFrame(X_tfidf.toarray())], axis=1)
print("\n\nTF-IDF:")
print(X_tfidf_feat)

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.multioutput import MultiOutputRegressor
#
#
# #rf = RandomForestClassifier()
#
# rf_regressor = RandomForestRegressor(random_state=0)
# multi_output_regressor = MultiOutputRegressor(rf_regressor)
#
# param = {'n_estimators': [10, 150, 300],
#         'max_depth': [30, 60, 90, None]}
#
#
# pd.set_option('display.max_columns', None) #wyswietla wszystkie kolumny
#
# gs = GridSearchCV(multi_output_regressor, param, cv=5, n_jobs=-1)
# X_tfidf_feat.columns = X_tfidf_feat.columns.astype(str)
# # y = pd.concat([myDataset['pos'], myDataset['neg']], axis=1)
# gs_fit = gs.fit(X_tfidf_feat, myDataset['pos'])
# print("\n\nTF-IDF test scores:")
# print(pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5])

import matplotlib.pyplot as plt
import numpy as np

# Histogram for 'body_len'
bins = np.linspace(0, myDataset['body_len'].max(), 40)
plt.hist(myDataset[myDataset['score'] == 'positive']['body_len'], bins, alpha=0.5, label='positive')
plt.hist(myDataset[myDataset['score'] == 'negative']['body_len'], bins, alpha=0.5, label='negative')
plt.legend(loc='upper right')
plt.title("Body Length Distribution")
plt.show()

# Histogram for 'punct%'
bins = np.linspace(0, (myDataset['punct%'].max())**(1/5), 10)
plt.hist((myDataset[myDataset['score'] == 'positive']['punct%'])**(1/5), bins, alpha=0.5, label='positive')
plt.hist((myDataset[myDataset['score'] == 'negative']['punct%'])**(1/5), bins, alpha=0.5, label='negative')
plt.legend(loc='upper right')
plt.title("Punctuation % Distribution")
plt.show()

# Histogram for 'CAPS%'
bins = np.linspace(0,(myDataset['CAPS%'].max())**(1/5), 10)
plt.hist((myDataset[myDataset['score'] == 'positive']['CAPS%'])**(1/5), bins, alpha=0.5, label='positive')
plt.hist((myDataset[myDataset['score'] == 'negative']['CAPS%'])**(1/5), bins, alpha=0.5, label='negative')
plt.legend(loc='upper right')
plt.title("Capital Letters % Distribution")
plt.show()

# Histogram for 'body_len'
bins = np.linspace(0, myDataset['body_len'].max(), 40)
plt.hist(myDataset['body_len'], bins)
plt.title("Body Length Distribution")
plt.show()

# Histogram for 'punct%'
bins = np.linspace(0, myDataset['punct%'].max(), 40)
plt.hist(myDataset['punct%'], bins)
plt.title("Punctuation % Distribution")
plt.show()

# Histogram for 'CAPS%'
bins = np.linspace(0, myDataset['CAPS%'].max(), 40)
plt.hist(myDataset['CAPS%'], bins)
plt.title("Capital Letters % Distribution")
plt.show()

# # Box-Cox transformations for 'body_len'
# for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
#     plt.hist((myDataset['body_len'])**(1/i), bins=40)
#     plt.title("Box Cox transformation(body_len): 1/{}".format(str(i)))
#     plt.show()
#
# # Box-Cox transformations for 'punct%'
# for i in [1, 2, 3, 4, 5]:
#     plt.hist((myDataset['punct%'])**(1/i), bins=40)
#     plt.title("Box Cox transformation(punct%): 1/{}".format(str(i)))
#     plt.show()
#
# for i in [1, 2, 3, 4, 5]:
#     plt.hist((myDataset['CAPS%'])**(1/i), bins=40)
#     plt.title("Box Cox transformation(CAPS%): 1/{}".format(str(i)))
#     plt.show()


from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

rf_regressor = RandomForestRegressor(random_state=0)

multi_output_regressor = MultiOutputRegressor(rf_regressor)

param = {'estimator__n_estimators': [300],
        'estimator__max_depth': [90]}

pd.set_option('display.max_columns', None)

gs = GridSearchCV(multi_output_regressor, param, cv=5, n_jobs=-1)
print("1")
X_tfidf_feat.columns = X_tfidf_feat.columns.astype(str)
print("2")
y = pd.concat([myDataset['pos'], myDataset['neg']], axis=1)
print("3")
gs_fit = gs.fit(X_tfidf_feat, y)
print("\n\nTF-IDF test scores:")
print(pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5])

import pickle

# Save the trained model as a pickle string.
saved_model = pickle.dumps(gs_fit)

# Save the fitted TfidfVectorizer
with open('tfidf_vect_all2.pkl', 'wb') as file:
    pickle.dump(tfidf_vect, file)

# Save the fitted model
with open('saved_model_all2.pkl', 'wb') as file:
    pickle.dump(gs_fit, file)

def make_prediction(input_text):
    # Prepare the input
    input_data = pd.DataFrame([input_text], columns=['sentence'])
    input_data['body_len'] = input_data['sentence'].apply(lambda x: len(x) - x.count(" "))
    input_data['punct%'] = input_data['sentence'].apply(lambda x: count_punct(x))
    input_data['CAPS%'] = input_data['sentence'].apply(lambda x: len([x for x in x.split() if x.isupper()]) / len(x.split()) * 100 if len(x.split()) != 0 else 0)

    # Transform the input
    X_input_tfidf = tfidf_vect.transform(input_data['sentence'])
    X_input_tfidf_feat = pd.concat([input_data['body_len'], input_data['punct%'], input_data['CAPS%'], pd.DataFrame(X_input_tfidf.toarray())], axis=1)

    # Make sure the columns are of type string
    X_input_tfidf_feat.columns = X_input_tfidf_feat.columns.astype(str)

    # Make the prediction
    prediction = gs_fit.predict(X_input_tfidf_feat)

    return prediction

# Use the function
input_text = input("Please enter your input text: ")
prediction = make_prediction(input_text)
print(prediction)