import json
import time
from bs4 import BeautifulSoup
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
import diskcache as dc

app = Flask(__name__)
CORS(app)

cache = dc.Cache('cache_directory')

#database setup during the first run
import sqlite3

def init_db():
    conn = sqlite3.connect('../NextJS/nextjs_front/prisma/scrapper.db')
    if conn is None:
        print("Error connecting to the database")
        return
    cursor = conn.cursor()
    ############################
    # Tutaj UNIQUE moze narobić dużo problemów więc warto na to uważać
    ############################
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Company (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            url TEXT,
            review_count INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db()

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

def update_company_data(name, url, review_count):
    conn = sqlite3.connect('../NextJS/nextjs_front/prisma/scrapper.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO Company (name, url, review_count)
        VALUES (?, ?, ?)
    ''', (name, url, review_count))
    conn.commit()
    conn.close()


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

        review_count = len(reviews[name_of_company])
        update_company_data(name_of_company, url, review_count.__int__())

        print(reviews)
        return reviews


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
            reviews = business['numberOfReviews']
            global_names.append(name)
            global_hrefs.append(href)
            print(name)
            print(href)
            print(reviews)
            conn = sqlite3.connect('../NextJS/nextjs_front/prisma/scrapper.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO Company (name, url, review_count) VALUES (?, ?, ?)
            ''', (name, href, reviews))
            conn.commit()
            conn.close()


# trustpilot
# pobieranie ilosci stron

def scrape_all_companies():
    scrapeCategories("bank")
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

scrape_all_companies()

@app.route('/reviews/<name>', methods=['GET'])
def get_reviews(name):
    cached_data = cache.get(name)
    if cached_data:
        return jsonify(cached_data)
    
    reviews = scrape(name)
    cache.set(name, reviews, expire=3600)  # Cache for 1 hour
    return jsonify(reviews)

@app.route('/companies', methods=['GET'])
def get_companies():
    conn = sqlite3.connect('../NextJS/nextjs_front/prisma/dev.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, review_count FROM Company')
    companies = cursor.fetchall()
    conn.close()
    return jsonify(companies)


@app.route('/rescrape', methods=['POST'])
def rescrape():
    scrape_all_companies()
    cache.clear()  # Clear the cache after rescraping
    return jsonify({"message": "Rescraping started"}), 200  # Ensure a valid response tuple is returned

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Runs the Flask server accessible from other devices

