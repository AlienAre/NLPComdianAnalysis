#Web scraping, pickle imports
import requests
import pickle
from bs4 import BeautifulSoup

url = "http://scrapsfromtheloft.com/2017/05/06/louis-ck-oh-my-god-full-transcript/"

page = requests.get(url).text.encode("utf-8")
#print(page.encode("utf-8"))
soup = BeautifulSoup(page, "lxml")
text = [p.text for p in soup.find(class_="post-content").find_all("p")]
#print(len(text))
print(text)
