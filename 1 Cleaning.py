#Web scraping, pickle imports
import requests
import pickle
from bs4 import BeautifulSoup

def url_to_transcript(url):
    page = requests.get(url).text
    #page = requests.get(url).text.encode("utf-8") #need to encode so it can get the correct text
    soup = BeautifulSoup(page, "lxml")
    text = [p.text for p in soup.find(class_="post-content").find_all("p")]
    print(url)
    #print(text)
    return text

# URLs of transcripts in scope
urls = ['http://scrapsfromtheloft.com/2017/05/06/louis-ck-oh-my-god-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/11/dave-chappelle-age-spin-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/15/ricky-gervais-humanity-transcript/',
        'http://scrapsfromtheloft.com/2017/08/07/bo-burnham-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/05/24/bill-burr-im-sorry-feel-way-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/04/21/jim-jefferies-bare-2014-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/02/john-mulaney-comeback-kid-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2017/10/21/hasan-minhaj-homecoming-king-2017-full-transcript/',
        'http://scrapsfromtheloft.com/2017/09/19/ali-wong-baby-cobra-2016-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/03/anthony-jeselnik-thoughts-prayers-2015-full-transcript/',
        'http://scrapsfromtheloft.com/2018/03/03/mike-birbiglia-my-girlfriends-boyfriend-2013-full-transcript/',
        'http://scrapsfromtheloft.com/2017/08/19/joe-rogan-triggered-2016-full-transcript/']

# Comedian names
comedians = ['louis', 'dave', 'ricky', 'bo', 'bill', 'jim', 'john', 'hasan', 'ali', 'anthony', 'mike', 'joe']

#print(requests.get('https://hcm17.sapsf.com/sf/orgchart?selected_user=2010107&_s.crb=MmEGS8UPkzZc%2fNNO4CXmbAwacXw%3d').text)

##for tesst use
#url = ['http://scrapsfromtheloft.com/2017/05/06/louis-ck-oh-my-god-full-transcript/']
#person = ['louis']
#trs = url_to_transcript('http://scrapsfromtheloft.com/2017/05/06/louis-ck-oh-my-god-full-transcript/')

## Actually request transcripts (takes a few minutes to run)
#transcripts = [url_to_transcript(u) for u in urls]

##Pickle files for later use

##Make a new directory to hold the text files
#!mkdir transcripts

#for i, c in enumerate(comedians):
#    with open("MLtest\\NLPComdianAnalysis\\transcripts\\" + c + ".txt", "wb") as file:
#       pickle.dump(transcripts[i], file)

##Load pickled files
data = {}

for i, c in enumerate(comedians):
    with open("MLtest\\NLPComdianAnalysis\\transcripts\\" + c + ".txt", "rb") as file:
        #print(file.read())
        data[c] = pickle.load(file)

#check
#print(data.keys())

#More checks
#print(data['louis'][:2])
#print(len(data['louis']))

#Let's take a look at our data again
#print(next(iter(data.keys())))

#We are going to change this to key: comedian, value: string format
def combine_text(list_of_text):
    '''Takes a list of text and combines them into one large chunk of text.'''
    combined_text = ' '.join(list_of_text)
    return combined_text

#Combine value to single text
data_combined = {key: [combine_text(value)] for (key, value) in data.items()}
#print(data_combined)

import pandas as pd
pd.set_option('max_colwidth',150)

df_data = pd.DataFrame.from_dict(data_combined).transpose()
df_data.columns = ["transcript"]
df_data = df_data.sort_index()
#print(df_data.head())

##Let's take a look at the transcript for Ali Wong
#print(df_data["transcript"]["ali"]) #not use print due to unicode

##Apply a first round of cleaning
import re, string

def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text

round1 = lambda x: clean_text_round1(x)

##Let's take a look at the updated text
data_clean = pd.DataFrame(df_data["transcript"].apply(round1))

##Apply a second round of cleaning
def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub("[‘’“”…]", "", text)
    text = re.sub("\n", "", text)
    return text

round2 = lambda x: clean_text_round2(x)

##Let's take a look at the updated text
data_clean = pd.DataFrame(data_clean.transcript.apply(round2))

##Let's add the comedians' full names as well
full_names = ['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']

df_data['full_name'] = full_names

##Let's pickle it for later use
df_data.to_pickle("MLtest\\NLPComdianAnalysis\\files\\corpus.pkl")

##We are going to create a document-term matrix using CountVectorizer, and exclude common English stop words
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words="english")
cv_data = cv.fit_transform(data_clean["transcript"])
#print(cv.get_feature_names())
#print(cv_data)
dtm_data = pd.DataFrame(cv_data.toarray(), columns=cv.get_feature_names())
dtm_data.index = data_clean.index

dtm_data.to_pickle("MLtest\\NLPComdianAnalysis\\files\\dtm.pkl")

##Let's also pickle the cleaned data (before we put it in document-term matrix format) and the CountVectorizer object
data_clean.to_pickle("MLtest\\NLPComdianAnalysis\\files\\data_clean.pkl")
pickle.dump(cv, open("MLtest\\NLPComdianAnalysis\\files\\cv.pkl", "wb"))