import pandas as pd
import numpy as np
import math
from textblob import TextBlob
import matplotlib.pyplot as plt

##split text into pieces, each piece contains n char
def split_text(text, n=10):
    #Takes in a string of text and splits into n equal parts, with a default of 10 equal parts.
    #Calculate length of text, the size of each chunk of text and the starting points of each chunk of text
    length = len(text)
    size = math.floor(length / n)
    start = np.arange(0, length, size)

    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece]+size])
    return split_list

##read the file to analysis
data = pd.read_pickle("files\\corpus.pkl")
#print(data)

##Create quick lambda functions to find the polarity and subjectivity of each routine
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

data["polarity"] = data["transcript"].apply(pol)
data["subjectivity"] = data["transcript"].apply(sub)

#print(data)

'''
##Plot the data
plt.clf()
plt.rcParams["figure.figsize"] = [10,8]

for index, comedian in enumerate(data.index):
    x = data.polarity.loc[comedian]
    y = data.subjectivity.loc[comedian]
    plt.scatter(x, y, color="blue")
    plt.text(x+.001, y+.001, data['full_name'][index], fontsize=10)
    plt.xlim(-.01, .12) 
    
plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

plt.show()
'''
##split all transcripts
list_pieces = []
for t in data.transcript: 
    list_pieces.append(split_text(t))

print(len(list_pieces))
#print(list_pieces)

##Calculate the polarity for each piece 
polarity_transcript = []
for lp in list_pieces:
    polarity_piece = []
    for p in lp:
        polarity_piece.append(TextBlob(p).sentiment.polarity)
    polarity_transcript.append(polarity_piece)

#print(polarity_transcript)

##Show the plot for one comedian
plt.clf()
plt.plot(polarity_transcript[0])
plt.title(data["full_name"].index[0])
plt.show()

##Show the plot for all
plt.clf()
plt.rcParams['figure.figsize'] = [16, 12]

for index, comedian in enumerate(data.index):    
    plt.subplot(3, 4, index+1)
    plt.plot(polarity_transcript[index])
    plt.plot(np.arange(0,10), np.zeros(10))
    plt.title(data['full_name'][index])
    plt.ylim(ymin=-.2, ymax=.3)
    
plt.show()
