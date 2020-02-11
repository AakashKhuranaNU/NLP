import numpy
import spacy
import re
import nltk.stem
import matplotlib.pyplot as plt
from datetime import datetime
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import StanfordNERTagger
from nltk.corpus import stopwords
from collections import Counter

from textblob import TextBlob
import json
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nlp = spacy.load('en_core_web_sm')
t=[]
mul_categ = []
new_categ=[]
# stanford_ner_tagger = StanfordNERTagger(
#     'stanford_ner/' + 'classifiers/english.muc.7class.distsim.crf.ser.gz',
#     'stanford_ner/' + 'stanford-ner-3.9.2.jar'
# )

lem=WordNetLemmatizer()
genre_lis=["director","actor","actress","award","screenplay","motion","picture","tv","television","series","drama","comedy,musical"]
# print(lem.lemmatize("won"))
# print(lem.lemmatize("winning"))
# print(lem.lemmatize("winner"))
i=0
j=0
k=0
m=0
n=0
name=""
host=""
cont=0
lis=[]
c=1
estop = ['for','at',"https",'golden','http', '#', '.', '!','-', '?','\\', ':', ';', '"', "'",'the','but','although','#goldenglobes','and','`','who','&']
if c==1:

    row=open('C:\\Users\\khaak\\Documents\\gg2013.json', encoding="utf8")
    ti=json.load(row)
    # print(t['text'])
    time=[]
    y=[]
    count=0
    for t in ti:
        # print("hey",t["timestamp_ms"])
        ts=t["timestamp_ms"]
        # dt_object = datetime.fromtimestamp(ts)
        time.append(ts)
        # print("dt_object =", dt_object)
        # print("type(dt_object) =", type(dt_object))
        str=t["text"].replace("Best","")
        blob = TextBlob(str)
        y.append(blob.sentences[0].sentiment.polarity)
        count+=1
        print(count)
        # print(y)
    int=time[0]
    count=0
    sum1=0
    sum2=0
    x=0
    xax=[]
    yax=[]
    for i in range(0,len(time)):
        diff=time[i]-int
        print(diff)
        print("count",count)
        if float(diff)>5000:
            print("change",diff," ",count)
            x=x+5000
            y1=(sum2/count)
            xax.append(x)
            yax.append(y1)
            int=time[i]
            count=1
            sum1=0
            sum2=0
        else:
            sum1=sum1+time[i]
            sum2=sum2+y[i]
            count+=1
    l=len(y)
    print(xax)
    print(yax)
    plt.plot(xax,yax)
    plt.show()
print("total tweets",n,m)
