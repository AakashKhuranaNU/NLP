import numpy
import spacy
import re
import nltk.stem
from datetime import datetime
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import StanfordNERTagger
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt

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
has=[]
c=1
estop = ['for','at',"https",'golden','http', '#', '.', '!','-', '?','\\', ':', ';', '"', "'",'the','but','although','#goldenglobes','and','`','who','&']
if c==1:

    row=open('C:\\Users\\khaak\\Documents\\gg2013.json', encoding="utf8")
    ti=json.load(row)
    # print(t['text'])
    for t in ti:
        # print("hey",t["timestamp_ms"])
        ts=t["timestamp_ms"]/1000
        s=t["text"].split(" ")
        for i in s:
            if '#' in i:
                i=i.lower()
                if "golden" not in i[1:]:
                    has.append(i)
                    print(i[1:])

    ed=Counter(has).most_common(30)
    x=[]
    y=[]
    for i in ed:
        # print(i[0])
        x.append(i[0])
        y.append(i[1])


    # print(y)
    # plt.plot(x,y)
    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    # langs = x
    # students = y
    # ax.bar(langs, students)
    #
    # # plt.show()
    # plt.xlabel('Hashtags')
    # # naming the y axis
    # plt.ylabel('Frequency')
    # # giving a title to my graph
    # plt.title('Hashtags Trend')
    # plt.show()

    plt.bar(x, y)
    # plt.yticks(y, x)

    plt.xticks(rotation=90)
    plt.ylabel('Frequency')
    plt.xlabel('Hashtags')
    plt.title('Hashtags Trend')
    plt.show()
print("total tweets",n,m)
