from bs4 import BeautifulSoup
import requests

def scrape_website(url,headers=None):
    page = requests.get(url,headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup

url = "https://diecezja.rzeszow.pl/parafie/"
first_page = scrape_website(url)
number_of_pages = first_page.find_all('a', attrs={"class": "page-link"})
number_of_pages = number_of_pages[number_of_pages.__len__()-2].get_text()
print(number_of_pages)


global_names = []
global_hrefs = []

#pobieranie nazw i linkow do firm
for i in range(1,int(number_of_pages)):
    url = "https://diecezja.rzeszow.pl/parafie/page/{}/".format(i)
    page = scrape_website(url)
    #pobiernie nazw firm
    articles = page.find_all('article', attrs={"class": "my-5"})
    names = [name.find('h2').get_text().lower() for name in articles]
    #pobieranie linkow do firm
    link_elements = [name.find('a') for name in articles]
    hrefs = [link.get('href') for link in link_elements]
    print(names)
    print(hrefs)
    print("\nend of page {}\n\n".format(i))
    global_names.extend(names)
    global_hrefs.extend(hrefs)


