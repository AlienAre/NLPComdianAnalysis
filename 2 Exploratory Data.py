import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_pickle("files\\dtm.pkl")
data = data.transpose()
#print(data.head())
#print(data.index)

##Find the top 30 words said by each comedian
top_dict = {}
for c in data.columns:
    top = data[c].sort_values(ascending=False).head(30)
    top_dict[c] = list(zip(top.index, top.values))

#print(top_dict)
##Print the top 15 words said by each comedian
#for comedian, top_words in top_dict.items():
#    print(comedian)
#    print(", ".join([word for word, count in top_words[:14]]))
#    print("------")

##Add common top words to stop word list
from collections import Counter

words = []
for comedian in data.columns: 
    top = [word for (word, count) in top_dict[comedian]]
    for t in top: 
        words.append(t)

##Let's aggregate this list and identify the most common words along with how many routines they occur in
#print(Counter(words).most_common())

##If more than half of the comedians have it as a top word, exclude it from the list
add_stop_words = [word for word, count in Counter(words).most_common() if count > 6]

##Let's update our document-term matrix with the new list of stop words
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

##Read cleaned data
data_clean = pd.read_pickle("files\\data_clean.pkl")

##Add new stop words
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)


##Recreate document-term matrix
cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean["transcript"])
#print(data_cv)
#print(type(data_cv))
data_stop = pd.DataFrame(data_cv.toarray(),columns=cv.get_feature_names())
data_stop.index = data_clean.index

##Pickle it for later use
import pickle
pickle.dump(cv, open("files\\cv_stop.pkl", "wb"))
data_stop.to_pickle("files\\dtm_stop.pkl")

'''######################
need to fix error: Microsoft Visual C++ 14.0 is required. to make following works
##make some word clouds
from worldcloud import WordCloud

wc = WordCloud(stopwords=stop_words, backgroud_color="white", colormap="Dark2", max_font_size=100, random_state=42)

##reset the output dimensions

plt.rcParams['figure.figsize'] = [16, 6]

full_names = ['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']

# Create subplots for each comedian
for index, comedian in enumerate(data.columns):
    wc.generate(data_clean.transcript[comedian])
    
    plt.subplot(3, 4, index+1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(full_names[index])
    
plt.show()
####################'''

##Identify the non-zero items in the document-term matrix, meaning that the word occurs at least once
#print(data["ali"])
#print(data["ali"].nonzero()[0])
#print(data["ali"].nonzero()[0].size)
full_names = ['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']
unique_list = []
for comedian in data.columns: 
    uniques = data[comedian].nonzero()[0].size
    unique_list.append(uniques)

##Create a new dataframe that contains this unique word count
data_words = pd.DataFrame(list(zip(full_names, unique_list)), columns=["comedian","unique_words"])
data_unique_sort = data_words.sort_values(by="unique_words")
#print(data_unique_sort)

##Calculate the words per minute of each comedian

##Find the total number of words that a comedian uses
total_list =[]
for comedian in data.columns:
    total = sum(data[comedian])
    total_list.append(total)

##Comedy special run times from IMDB, in minutes
run_times = [60, 59, 80, 60, 67, 73, 77, 63, 62, 58, 76, 79]

data_words["total_words"] = total_list
data_words["run_times"] = run_times
data_words["word_per_mins"] = data_words["total_words"] / data_words["run_times"]

data_wpm_sort = data_words.sort_values(by="word_per_mins")
#print(data_wpm_sort)

##plot the data
plt.clf()
y_pos = np.arange(len(data_words))

plt.subplot(1,2,1)
plt.barh(y_pos, data_unique_sort["unique_words"], align="center")
plt.yticks(y_pos, data_unique_sort["comedian"])
plt.title("Unique Words Count", fontsize=20)

plt.subplot(1,2,2)
plt.barh(y_pos, data_wpm_sort["word_per_mins"], align="center")
plt.yticks(y_pos, data_wpm_sort["comedian"])
plt.title("Words per Mins", fontsize=20)

plt.tight_layout()
#plt.show()

##Analysis: Amount of Profanity
Counter(words).most_common()

##get bad words
data_bad_words = data.transpose()[["fucking", "fuck", "shit"]]
#print(data_bad_words)
data_profanity = pd.concat([data_bad_words.fucking + data_bad_words.fuck, data_bad_words.shit], axis=1)
data_profanity.columns = ["f_word", "s_word"]
print(data_profanity)


##Let's create a scatter plot of our findings
plt.clf()
plt.rcParams['figure.figsize'] = [10, 8]

for i, comedian in enumerate(data_profanity.index):
    x = data_profanity.f_word.loc[comedian]
    y = data_profanity.s_word.loc[comedian]
    plt.scatter(x, y, color='blue')
    plt.text(x+1.5, y+0.5, full_names[i], fontsize=10)
    plt.xlim(-5, 155) 
    
plt.title('Number of Bad Words Used in Routine', fontsize=20)
plt.xlabel('Number of F Bombs', fontsize=15)
plt.ylabel('Number of S Words', fontsize=15)

plt.show()