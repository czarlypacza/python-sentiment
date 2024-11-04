import json
import time
from bs4 import BeautifulSoup
import requests

global_names = []
global_hrefs = []


def scrape_website(url, headers=None):
    page = requests.get(url, headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup


def scrape_reviews(review_page, name_of_company, reviews):
    review_elements = review_page.find_all('div', attrs={"data-review-content": "true"})

    for review_element in review_elements:
        # Scrape the title and text of the review
        title = review_element.find('h2').get_text()
        text = review_element.find_all('p')[0].get_text()

        # Create a dictionary for the review
        review = {"title": title, "text": text}

        # Append the review to the list associated with the company
        reviews[name_of_company].append(review)

    return reviews


def scrape(name):
    name_of_company = name.lower()
    matched_names = [name for name in global_names if name_of_company in name]
    print(matched_names)

    if len(matched_names) == 1:
        global_names_index = global_names.index(matched_names[0])
        url = global_hrefs[global_names_index]
        print(url)

        review_page = scrape_website(url)

        # Find the number of pages
        pagination_buttons = review_page.find_all('a',
                                                  attrs={"aria-label": lambda x: x and x.startswith("Page number")})
        number_of_pages = 1  # Default to 1 if not found
        if pagination_buttons:
            number_of_pages = int(pagination_buttons[-1].get_text())
        print(number_of_pages)

        reviews = {}
        reviews[name_of_company] = []

        reviews = scrape_reviews(review_page, name_of_company, reviews)

        for i in range(2, number_of_pages + 1):
            url = f"{global_hrefs[global_names_index]}?page={i}"
            review_page = scrape_website(url)
            reviews = scrape_reviews(review_page, name_of_company, reviews)
            print(f"page {i}")

        print(reviews)
        return reviews


# from sklearn.feature_extraction.text import TfidfVectorizer
# import pandas as pd
#
# myDataset = pd.read_json("results(4).json", orient="records", lines=True)
# print(myDataset)


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


from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def scrapeCategories(name):
    url = f"https://www.trustpilot.com/categories/{name}?page=1"
    first_page = scrape_website(url)

    # Find the number of pages
    number_of_pages = 1  # Default to 1 if not found
    pagination_button = first_page.find('a', attrs={"name": "pagination-button-last"})
    if pagination_button:
        number_of_pages = int(pagination_button.get_text())

    # Find the script tag containing the JSON data
    script_tag = first_page.find('script', {'id': '__NEXT_DATA__'})
    if script_tag:
        json_data = json.loads(script_tag.string)

        # Extract business names and links from the first page
        extract_business_info(json_data)

        # Scrape the remaining pages
        for i in range(2, number_of_pages + 1):
            time.sleep(5)
            url = f"https://www.trustpilot.com/categories/{name}?page={i}"
            page = scrape_website(url)
            script_tag = page.find('script', {'id': '__NEXT_DATA__'})
            if script_tag:
                json_data = json.loads(script_tag.string)
                extract_business_info(json_data)
            print(f"\nend of page {i}\n\n")


def extract_business_info(json_data):
    if 'pageProps' in json_data['props'] and 'businessUnits' in json_data['props']['pageProps']:
        business_units = json_data['props']['pageProps']['businessUnits']['businesses']
        for business in business_units:
            name = business['displayName'].lower()
            href = f"https://www.trustpilot.com/review/{business['identifyingName']}"
            global_names.append(name)
            global_hrefs.append(href)
            print(name)
            print(href)


# trustpilot
# pobieranie ilosci stron

#scrapeCategories("bank")
#scrapeCategories("car_dealer")
#scrapeCategories("jewelry_store")
#scrapeCategories("travel_insurance_company")
#scrapeCategories("furniture_store")
#scrapeCategories("clothing_store")
#scrapeCategories("fitness_and_nutrition_service")

#scrapeCategories("insurance_agency")

# scrapeCategories("mortgage_broker")

#scrapeCategories("real_estate_agents")
# scrapeCategories("womens_clothing_store")


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


# # Use the function
# input_text = input("Please enter your input text: ")
# prediction = make_prediction(input_text)
# print(prediction)


@app.route('/reviews/<name>', methods=['GET'])
def get_reviews(name):
    # Assuming reviews is your dictionary containing the review data
    return jsonify(scrape(name))


@app.route('/sentiment/<name>', methods=['GET'])
def classify_reviews(name):
    result = []
    reviews = scrape(name)
    for review in reviews[name]:
        for sentence in review['text'].split('.'):
            sentiment = make_prediction(sentence)
            sentiment = sentiment.tolist()
            result.append({sentence: sentiment})

    result = {'result': result}
    return jsonify(result)

@app.route('/sentiment', methods=['POST'])
def classify_review():
    data = request.get_json()
    input_text = data.get('text', '')

    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    sentences = input_text.split('.')
    results = []
    scores = []

    for sentence in sentences:
        if sentence.strip():
            sentiment = make_prediction(sentence)
            sentiment = sentiment.tolist()
            pos, neg = sentiment[0]
            sentiment_label = "positive" if pos > neg else "negative" if neg > pos else "neutral"
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
    app.run(host='0.0.0.0', port=5000)  # Runs the Flask server accessible from other devices

#/review/www.allianztravelinsurance.com
#/review/withfaye.com
# url = "https://www.trustpilot.com/review/www.flashbay.com?page=2"
# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
# }
#
# write_file = open("trustpilot.html", "w")
# #write_file.write(str(scrape_website(url,headers).encode('utf-8')))
#
#
# soup = scrape_website(url,headers)
# p_elements = soup.find_all('p')
# p_text = [p.get_text() for p in p_elements]
#
# for p in p_text:
#     print(p+"\n\n")


# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
#
# options = Options()
# options.headless = True
# options.add_argument("--window-size=1920,1200")
#
# # Initialize the WebDriver with options
# driver = webdriver.Chrome(options=options)
#
# # Navigate to the page and scrape data
# driver.get("https://twitter.com/search?q=Mcdonalds")
# write_file = open("twitter.html", "w")
# write_file.write(driver.page_source.encode('utf-8'))
#
# # Close the browser
# driver.quit()
