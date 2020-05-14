##Read document-term matrix
import pandas as pd
import pickle
from gensim import matutils, models
import scipy.sparse
#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data = pd.read_pickle("files\\dtm_stop.pkl")
#print(data)

tdm = data.transpose()
#print(tdm.head())

##We're going to put the term-document matrix into a new gensim format, from df --> sparse matrix --> gensim corpus
sparse_counts = scipy.sparse.csr_matrix(tdm)
#print(sparse_counts)
corpus = matutils.Sparse2Corpus(sparse_counts)

##Gensim also requires dictionary of the all terms and their respective location in the term-document matrix
cv = pickle.load(open("files\\cv_stop.pkl", "rb"))
id2word = dict((v, k) for k, v in cv.vocabulary_.items())
#print(cv)
#print(id2word)

##Now that we have the corpus (term-document matrix) and id2word (dictionary of location: term),
##we need to specify two other parameters as well - the number of topics and the number of passes
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=2, passes=10)
print(lda.print_topics())
lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=3, passes=10)
print(lda.print_topics())
