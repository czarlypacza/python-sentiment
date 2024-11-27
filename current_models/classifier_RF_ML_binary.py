import re
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.corpus import stopwords as sw
import string
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV

lemmatizer = WordNetLemmatizer()

with open("global_data_binary.json", "r") as file:
    data = json.load(file)


# Initialize an empty list to store all reviews
all_reviews = []

# Iterate through each rating (negative, positive)
for rating in ["positive", "negative"]:
    if rating in data:
        for review in data[rating]:
            all_reviews.append(review)
            
print(all_reviews[len(all_reviews)-5:])

# Convert to DataFrame
myDataset = pd.DataFrame(all_reviews)
myDataset['sentence'] = myDataset['title'] + ' ' + myDataset['text']

#yDataset = myDataset[0:100000]
#print(myDataset)
print(myDataset.head(10))


#Replace"Date of experience: <some date> (example: September 26, 2024)" with ""
myDataset['sentence'] = myDataset['sentence'].apply(lambda x: re.sub(r'Date of experience: [A-Za-z]+\s\d{1,2},\s\d{4}', '', x))

# Remove rows where 'sentence' is an empty string
myDataset = myDataset[myDataset['sentence'] != '']

# Reset the index
myDataset = myDataset.reset_index(drop=True)

print(myDataset.shape)

stopwords = sw.words('english')


def count_punct(text):
    text_length = len(text) - text.count(" ")
    if text_length == 0:
        return 0
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / text_length, 3) * 100


myDataset['body_len'] = myDataset['sentence'].apply(lambda x: len(x) - x.count(" "))
myDataset['punct%'] = myDataset['sentence'].apply(lambda x: count_punct(x))
myDataset['CAPS%'] = myDataset['sentence'].apply(
    lambda x: len([x for x in x.split() if x.isupper()]) / len(x.split()) * 100 if len(x.split()) != 0 else 0)
myDataset['word_count'] = myDataset['sentence'].apply(lambda x: len(x.split()))
myDataset['all_Caps'] = myDataset['sentence'].apply(lambda x: len([x for x in x.split() if x.isupper()]))

#pd.set_option('display.max_columns', None)
print(myDataset.head(10))

### PREPROCESSING DANYCH ###
### ^^^^^^^^^^^^^^^^^^ ###



###\/\/\/\/\/\/\/\/\/\/###
### WEKTORYZACJA DANYCH ###

def clean_text(text):
    # First clean and tokenize the text
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    # Lemmatize and remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
    return " ".join(tokens)  # Return joined text instead of tokens

# Modified TF-IDF vectorizer
tfidf_vect = TfidfVectorizer(
    preprocessor=clean_text,  # Use clean_text as preprocessor instead of analyzer
    analyzer='word',  # Use default word analyzer
    max_features=None,
    ngram_range=(1, 2),  # Now ngrams will work
    min_df=5,
    max_df=0.95,
    tokenizer=None  # Let TfidfVectorizer handle tokenization
)
X_tfidf = tfidf_vect.fit_transform(myDataset['sentence'])
X_tfidf_feat = pd.concat(
    [myDataset['body_len'], myDataset['punct%'], myDataset['CAPS%'],myDataset['word_count'],myDataset['all_Caps'] , pd.DataFrame(X_tfidf.toarray())], axis=1)
print("\n\nTF-IDF:")
print(X_tfidf_feat.head(16))

### WEKTORYZACJA DANYCH ###
### ^^^^^^^^^^^^^^^^^^ ###


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV

rf_classifier = RandomForestClassifier(random_state=0, n_jobs=-1)

# Initialize StratifiedKFold
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


param = {
    'n_estimators': [300],  # Default is 100
    'criterion': ['gini'],  # Default is 'gini'
    'max_depth': [None],  # Default is None
    'min_samples_split': [2],  # Default is 2
    'min_samples_leaf': [1],    # Default is 1
    'min_weight_fraction_leaf': [0.0],  # Default is 0.0
    'max_features': [None],  # Default is 'sqrt'
    'max_leaf_nodes': [None],  # Default is None
}

pd.set_option('display.max_columns', None)

gs = GridSearchCV(
    estimator=rf_classifier,
    param_grid=param,
    cv=skf,
    n_jobs=-1,
    verbose=3,
    pre_dispatch=1,
    return_train_score=True
)



print("1")
X_tfidf_feat.columns = X_tfidf_feat.columns.astype(str)
print("2")
y = myDataset['rating'].astype(str)
print("3")
gs_fit = gs.fit(X_tfidf_feat, y)
print("\n\nTF-IDF test scores:")
scores = pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)
print(scores[0:5])
open("scores.txt", "w").write(str(scores))





import pickle
import json

# Save the trained model as a pickle string.
saved_model = pickle.dumps(gs_fit)

# Save the fitted TfidfVectorizer
with open('./model_RF_ML_binary/tfidf_vect_new.pkl', 'wb') as file:
    pickle.dump(tfidf_vect, file)

# Save the fitted model
with open('./model_RF_ML_binary/saved_model_new.pkl', 'wb') as file:
    pickle.dump(gs_fit, file)

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
    prediction = gs_fit.predict_proba(X_input_tfidf_feat)

    return prediction

loop = True
while loop:
    input_text = input("Please enter your input text: ")
    if input_text == "exit":
        loop = False
        break
    prediction = make_prediction(input_text)
    print(prediction)
