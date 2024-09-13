# import tweepy
#
# #auth = tweepy.OAuth1UserHandler('g64rrbobvyzEzh3AMGNI5lFIX', 'Uac6nh4RyoTIToYOY6PiLa2ucccKaiPH7GfJMr0nxcOQnZJpki','805419647092789249-fVQoawqENWUKkYuANLQgjHPqymNVuGS', '2T6ICC6VmGD9o2mht53XmNZuQMS7Q52nHInrNLjBg2774')
# #auth.set_access_token('805419647092789249-fVQoawqENWUKkYuANLQgjHPqymNVuGS', '2T6ICC6VmGD9o2mht53XmNZuQMS7Q52nHInrNLjBg2774')
# #bearer
# #"AAAAAAAAAAAAAAAAAAAAAOhEswEAAAAA4actBYiAnSMiSKmPitdLG5XntxE%3DZv1mQLYcRLwM1e3YrXGyP6t1NslPNWFanoVCiY1tqJzhGbVTBn"
#
# #oauth 2.0 cient_id and secret
# #client_id = "U29zMkQwTTlqblRIejZZTHJXdU46MTpjaQ"
# #client_secret = "Ui_AiokN1IvEkkMxB5Kd7gohAYsfCpJUPSbfgzpHskDQliaovJ"
#
# api = tweepy.Client(consumer_key='g64rrbobvyzEzh3AMGNI5lFIX', consumer_secret='Uac6nh4RyoTIToYOY6PiLa2ucccKaiPH7GfJMr0nxcOQnZJpki', access_token='805419647092789249-EyyQr2ERSSJOYKg53dzHwZJbZZctioh', access_token_secret='NxS1qgiPE5IUtlY0A8Rn5Ly7TWz175nPTOllGjbh6QfCE')
#
# # def search_tweets(query, count=5):
# #     for tweet in api.search_tweets(query, lang="pl",result_type= "recent", count=count):
# #         process_tweet(tweet)
# #
# # tweets = []
# #
# # def process_tweet(tweet):
# #     tweets.append(tweet)
# #     print(tweet.text+"\n")
# #
# # search_tweets("Szymon Hołownia", 5)
# # print("\n\n\n")
#
# tweets = api.create_tweet(text="Twitter api")
# print(tweets)

import requests
from bs4 import BeautifulSoup

url = "https://bindingofisaacrebirth.fandom.com/wiki/Necropolis?dlcfilter=0"
response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')


table = soup.find('table' )

rows = table.find_all('tr')
filtered_rows = [row for row in rows if not row.find('img', class_='dlc')]

for row in filtered_rows:
    td = row.find('td')
    if td is not None:

        print(td.find('a').text.replace(' ', '-').replace('\n','') + " found-in Depths.")

#
# filtered_rows = [row for row in rows if not row.find('img', class_='dlc')]
#
#
# for row in filtered_rows:
#     td = row.find('td')
#     if td is not None:
#         name = td.find('a').text.replace(' ', '-')
#         print(name.replace('\n','') + " is a monster.")












