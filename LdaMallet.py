#!/usr/bin/env python
import logging, gensim, bz2
from gensim import *
from gensim import models
from gensim.models import ldamulticore,LdaMallet
import csv
import nltk
from nltk import bigrams
from nltk import trigrams
from collections import Counter
dataset = []
with open('abstract_L50.txt','r') as f:
    for row in f:
        temp = row.translate(None,"\"").strip()
        dataset.append(temp)
print len(dataset)
# stopword
common = ['the','of','and','a','to','in','is','you','that','it',
           'he','was','for','on','are','as','with','his','they','i',
           'at','be','this','have','from','or','one','had','by','word',
           'but','not','what','all','were','we','when','your','can','said',
           'there','use','an','each','which','she','do','how','their','if',
           'will','up','other','about','out','many','then','them','these','so',
           'some','her','would','make','like','him','into','time','has','look',
           'two','more','write','go','see','number','no','way','could','people',
           'my','than','first','water','been','call','who','oil','its','now',
           'find','long','down','day','did','get','come','made','may','part','missing']

stoplist = set(common)

def get_token(data):
    tokens = []
    for document in data:
        temp = nltk.word_tokenize(document.decode('ascii','ignore').encode('ascii'))
        temp = [token.translate(None,".!?,;:*+=\)\(\[\]\\\n/'\"").replace('--','') for token in temp]
        temp = [token.lower() for token in temp if len(token) > 2]
        tokens.append(temp)
    return tokens

texts = get_token(documents)


print len(texts)
all_tokens = {}
for i in range(0,len(texts)):
    for j in range(0,len(texts[i])):
        if all_tokens.has_key(texts[i][j]):
            all_tokens[texts[i][j]] +=1
        else:
            all_tokens[texts[i][j]] = 1
for stop in stoplist:
    if all_tokens.has_key(stop):
        all_tokens.pop(stop,None)

tokens_once = []
for term in all_tokens.keys():
    if all_tokens[term] == 1:
        tokens_once.append(term)

new_texts = []

for text in texts:
    temp = []
    for word in text:
        if all_tokens.has_key(word):# and all_tokens[word] > 100:
            temp.append(word)
    new_texts.append(temp)

texts = new_texts
print len(texts)
dictionary = corpora.Dictionary(texts)
dictionary.save('temp/deerwester.dict')
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('temp/deerwester.mm', corpus)
tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]
print len(corpus)
lda = gensim.models.LdaMallet('~/project/mallet-2.0.7/bin/mallet',corpus=corpus,num_topics = 5,id2word=dictionary,workers=4, iterations = 1000)#, alpha=1.0/150)

topic2doc =  lda[corpus]

topic_list = lda.show_topics(num_topics=5, num_words=-1, formatted=False)
toCSV = []
for i in topic_list:
    temp = []
    for term in i:
        temp.append(term[0])
        temp.append(term[1].encode('ascii','ignore'))
    toCSV.append([])
    toCSV.append(temp)

with open("test_mallet.csv",'wb') as f:
    writer = csv.writer(f)
    writer.writerows(toCSV)

toCSV_t = []
for dist in topic2doc:
    temp = []
    for score in dist:
        temp.append(str(score[1]))
    toCSV_t.append(temp)
with open("topic_to_doc.csv",'wb') as f:
    writer = csv.writer(f)
    writer.writerows(toCSV_t)
print "DONE"
