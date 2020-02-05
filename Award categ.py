import numpy
import spacy
import re
import nltk.stem
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import StanfordNERTagger
from nltk.corpus import stopwords
from collections import Counter

import json
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nlp = spacy.load('en_core_web_sm')
t=[]
mul_categ = []
# stanford_ner_tagger = StanfordNERTagger(
#     'stanford_ner/' + 'classifiers/english.muc.7class.distsim.crf.ser.gz',
#     'stanford_ner/' + 'stanford-ner-3.9.2.jar'
# )

lem=WordNetLemmatizer()
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
if c==1:

    # row=open('C:\\Users\\khaak\\Documents\\gg2013.json', encoding="utf8")
    # ti=json.load(row)
    # # print(t['text'])
    # for t in ti:
    for row in open('C:\\Users\\khaak\\Documents\\gg2020.txt', encoding="utf8").readlines():
        t = json.loads(row)
        # i=t['text'].find("host")
        # print(t["text"])
        i1=re.search("best ",t['text'],re.IGNORECASE)
        i2 = re.search("for", t['text'], re.IGNORECASE)
        # i2=re.search("hosted",t['text'],re.IGNORECASE)
        l=t['text'].find("best")
        k+=1
        host=""
        count=0
        val=0
        if i1!=None :
            # doc = nlp(t['text'])
            # for ent in doc.ents:
            #     if(ent.label_=='PERSON'):
            #         print("the person identified")
            #         lis.append(ent.text)
            #
            #     print(ent.text, ent.start_char, ent.end_char, ent.label_)
            word_tokens = word_tokenize(t['text'])
            print("tokens",t["text"])
            count=0
            append=0
            categ=""

            for i in word_tokens:
                if i =="best":
                    append =1

                if append==1 and (i=='for') :
                    mul_categ.append(categ)
                    # print(t["text"])
                    # print("categ1", mul_categ[val])
                    val += 1
                    # print("categ",mul_categ)

                    categ=""
                    count=0
                    append=0

                if append == 1:
                    categ=categ+" "+i
                    count+=1

                if append==1 and count >15 :
                    categ = ""
                    count = 0
                    append = 0
            apeend=0
            count=0
            categ=""
            # print("multiple_categ",mul_categ)

            # print("CATEGO_1",t["text"])
        #     for pos in nltk.pos_tag(word_tokens):
        #         count+=1
        #         if pos[1] =="NNP":
        #             name=name+pos[0]
        #             cont=1
        #         elif cont==1 :
        #             cont=0
        #             host=host+name+"||||"
        #             name=''
        #         if re.search("host",pos[0],re.IGNORECASE):
        #             if pos[1][0:2] == "VB":
        #                 print("cnount now",count)
        #                 print("breaking now")
        #                 break
        #     print("count",count)
        #     # print("HOSTTTTTTTTTTTTTTTTTT",host)
        #
        #
        #     # print("sea",t['text'])
            n+=1
        #
        # if i!=-1:
        #     # print("============================================================================")
        #     # print("Host:",t['text'])
        #     # print("============================================================================")
        #     # word_tokens = word_tokenize(t['text'])
        #     # print("POS",nltk.pos_tag(word_tokens))
        #     j+=1
        #     # results = stanford_ner_tagger.tag(word_tokens)
        #     # print("NER", nltk.ne_chunk(nltk.pos_tag(word_tokens)))
        if l!=-1:
            # print("Best Only :",t['text'])
            m+=1
        if i2!=None:
            h=0
            # print("for :",t['text'])
            # m+=1
# print(lis)
# print(lis)
# print(Counter(lis))
print("multiple_categ",mul_categ)
print(Counter(mul_categ))
print("total tweets",n,m)
#     example_sent = "This is a sample sentence, showing off the stop words filtration."
#
#     stop_words = set(stopwords.words('english'))
#
#     word_tokens = word_tokenize(example_sent)
#
#     filtered_sentence = [w for w in word_tokens if not w in stop_words]
#
#     filtered_sentence = []
#
#     for w in word_tokens:
#         if w not in stop_words:
#             filtered_sentence.append(w)
#
#     print(word_tokens)
#     print(filtered_sentence)
#     # print(t)
# print("finish")
