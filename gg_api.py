'''Version 0.35'''
import json
import re
import collections
import nltk
from imdb import IMDb
from nltk import word_tokenize
from fuzzywuzzy import fuzz
import string
from nltk.corpus import stopwords
# from nltk.util import ngrams
import spacy
import pandas as pd
from nltk.stem import WordNetLemmatizer
import numpy as np
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nlp = spacy.load('en_core_web_sm')

OFFICIAL_AWARDS_LIST = []
CLEAN_AWARD_NAMES = {}
TWEETS = []
AWARD_TWEET_DICT = {}
AWARD_NOMINEE_DICT = {}
FILM_DATA = pd.DataFrame()
YEAR = 2013
stop_words = set(stopwords.words('english'))
FUZZ_LIMIT = 90


def get_tweets(YEAR):
    global TWEETS

    print("Loading tweets for year {}...".format(YEAR))
    f = 'gg{}.json'.format(YEAR)

    if YEAR == 2020:
        data = [json.loads(line) for line in open(f, 'r')]
        TWEETS = [t["text"].strip() for t in data]
        TWEETS = pre_process(TWEETS)
    else:
        fp = open(f, 'r')
        data = json.load(fp)
        TWEETS = [t["text"] for t in data]
        TWEETS = pre_process(TWEETS)
    print("Finished loading {} tweets".format(len(TWEETS)))

    generate_award_tweet_dict()
    return


def pre_process(tweets):
    arr = []
    for tweet in tweets:
        ind = tweet.find('#')
        if ind > -1:
            tweet = tweet[:ind]
        ind = tweet.find('http')
        if ind > -1:
            tweet = tweet[:ind]

        arr.append(tweet)

    return arr


def explore():
    count = collections.Counter([])
    for tweet in TWEETS:
        tweet = tweet.lower()
        if 'best' in tweet or 'award' in tweet:
            count.update(list(nltk.bigrams(nltk.tokenize.word_tokenize(tweet))))

    count = sorted(count.items(), key=lambda x: x[1], reverse=True)
    out_file = open('BOW_awards_bigram.txt', 'w', encoding='utf-8')
    for i in count:
        out_file.write(str(i[0]))
        out_file.write(' ')
        out_file.write(str(i[1]))
        out_file.write('\n')


def merge_results(arr, award):
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] and arr[j] and fuzz.token_sort_ratio(arr[i], arr[j]) >= FUZZ_LIMIT:
                arr[j] = ""

    arr = [s for s in arr if s]
    return arr
    # check_list = ['Motion Picture', 'Series', 'Film']
    #
    # if not award and not any(check.lower() in award for check in check_list):
    #     return arr
    #
    #
    # return new_arr


def generate_award_tweet_dict():
    global AWARD_TWEET_DICT, TWEETS, CLEAN_AWARD_NAMES, stop_words

    print("Processing Tweets...")

    '''
    cvec = CountVectorizer(stop_words='english')
    cvec.fit_transform(TWEETS)
    bow = list(cvec.vocabulary_.keys())
    sim_words = []
    syn1 = wordnet.synsets("television")[0]
    for word in bow:
        syn2 = wordnet.synsets(word)
        for syn in syn2:
            sim = syn1.wup_similarity(syn)
            if sim and (sim >= 0.8):
                sim_words.append(word)
                break

    print(sim_words)
    '''

    award_tweet_dict = {award: [] for award in OFFICIAL_AWARDS_LIST}
    stoplist = ['best', '-', 'award', 'for', 'or', 'made', 'in', 'a', 'by', 'performance', 'an', 'role', 'motion',
                'picture', 'television', 'limited', 'series', 'musical', 'comedy']
    clean_award_names = {award: [[a for a in award.lower().split(' ') if not a in stoplist]] for award
                         in OFFICIAL_AWARDS_LIST}

    # TODO: remove punctuation properly
    '''
    for award_names in clean_award_names.keys():
        for words in clean_award_names[award_names]:
            words.translate(str.maketrans('', '', string.punctuation))
    '''

    substitutes = {}
    substitutes["Limited Series or Motion Picture Made for Television"] = ['miniseries', 'mini-series', 'tv movie',
                                                                           'television movie', 'motion picture']
    substitutes["Motion Picture"] = ["picture", "movie", 'motion picture']
    substitutes["Television Series - Musical or Comedy"] = substitutes['Television Series - Comedy or Musical'] = [
        'television series', 'tv series', "tv comedy", "tv musical",
        "comedy series", "t.v. comedy",
        "t.v. musical", "television comedy", "television musical"]
    substitutes["Musical or Comedy"] = substitutes["Comedy or Musical"] = ['musical', 'comedy']
    substitutes["Series, Limited Series or Motion Picture Made for Television"] = substitutes[
        'Series, Miniseries or Motion Picture Made for Television'] = ['series', 'mini-series',
                                                                       'miniseries', 'limited series',
                                                                       'tv', 'television', 'tv movie',
                                                                       'tv series', 'television series',
                                                                       'motion picture']
    substitutes["Television Series - Drama"] = ['televisin series', 'tv series', "tv drama", "drama series",
                                                "television drama", "t.v. drama"]
    substitutes["Television Series"] = ['series', 'tv', 't.v.', 'television']
    substitutes["Television"] = ['tv', 't.v.']
    # substitutes["Foreign Language"] = ["foreign"]

    for award in OFFICIAL_AWARDS_LIST:
        for key in substitutes.keys():
            if key.lower() in award:
                clean_award_names[award].append(substitutes[key])
                break
        if len(clean_award_names[award]) < 2:
            clean_award_names[award].insert(0, [])

    CLEAN_AWARD_NAMES = clean_award_names

    for award in OFFICIAL_AWARDS_LIST:
        AWARD_TWEET_DICT[award] = []
        for tweet in TWEETS:
            a = all([word in tweet.lower() for word in clean_award_names[award][0]])
            b = any([word in tweet.lower() for word in clean_award_names[award][1]])
            if a and b:
                AWARD_TWEET_DICT[award].append(tweet)


def get_hosts(year):
    '''Hosts is a list of one or more strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here
    pre_ceremony(year)

    get_tweets(year)

    i = 0
    j = 0
    k = 0
    m = 0
    n = 0
    name = ""
    host = ""
    cont = 0
    lis = []
    # row = open('gg2013.json', encoding="utf8")
    # twwe = json.load(row)
    for tw in TWEETS:
        i = tw.find("host")
        # print(t["text"])
        i1 = re.search("hosting", tw, re.IGNORECASE)
        i2 = re.search("hosted", tw, re.IGNORECASE)
        l = tw.find("Ricky")
        k += 1
        host = ""
        count = 0
        if i1 != None or i2 != None:
            doc = nlp(tw)
            for ent in doc.ents:
                if (ent.label_ == 'PERSON'):
                    # print("the person identified")
                    lis.append(ent.text)

                # print(ent.text, ent.start_char, ent.end_char, ent.label_)
            word_tokens = word_tokenize(tw)
            # print("hosting",t["text"])
            for pos in nltk.pos_tag(word_tokens):
                count += 1
                if pos[1] == "NNP":
                    name = name + pos[0]
                    cont = 1
                elif cont == 1:
                    cont = 0
                    host = host + name + "||||"
                    name = ''
                if re.search("host", pos[0], re.IGNORECASE):
                    if pos[1][0:2] == "VB":
                        # print("cnount now",count)
                        # print("breaking now")
                        break

            # print("HOSTTTTTTTTTTTTTTTTTT",host)

            # print("sea",t['text'])
            n += 1

        if i != -1:
            # print("============================================================================")
            # print("Host:",t['text'])
            # print("============================================================================")
            # word_tokens = word_tokenize(t['text'])
            # print("POS",nltk.pos_tag(word_tokens))
            j += 1
            # results = stanford_ner_tagger.tag(word_tokens)
            # print("NER", nltk.ne_chunk(nltk.pos_tag(word_tokens)))
        if i == -1 and l != -1:
            # print("Ricky Only :",t['text'])
            m += 1
    print(lis)
    print(collections.Counter(lis))
    t = collections.Counter(lis)
    host_fil = []
    host_dic = {}
    disc = []
    t = t.most_common(7)
    print(t)
    for i in range(0, 7):
        ch = 0
        for j in range(0, 7):
            if (t[i][0] in t[j][0]):
                ch += 1
        if ch == 1:
            host_fil.append(t[i][0])
            host_dic[t[i][0]] = t[i][1]
        else:
            disc.append(t[i][0])
    fin = (collections.Counter(host_dic)).most_common(7)
    print("1", t)
    print("2", host_fil)
    print("3,", disc)
    print("4,", fin)
    ratio = (fin[1][1] / fin[0][1])
    ans = []
    ans.append(fin[0][0])
    if ratio > 0.65:
        ans.append(fin[1][0])
    print(ans)
    # return ans

    # print("ratio",(host_fil[0][1]/host_fil[1][1]))
    print("total tweets", j, k, m, n)
    # hosts = []
    return ans


def get_awards(year):
    '''Awards is a list of strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here

    t = []
    mul_categ = []
    new_categ = []
    # stanford_ner_tagger = StanfordNERTagger(
    #     'stanford_ner/' + 'classifiers/english.muc.7class.distsim.crf.ser.gz',
    #     'stanford_ner/' + 'stanford-ner-3.9.2.jar'
    # )

    lem = WordNetLemmatizer()
    genre_lis = ["director", "actor", "actress", "award", "screenplay", "motion", "picture", "tv", "television",
                 "series", "drama", "comedy,musical"]
    # print(lem.lemmatize("won"))
    # print(lem.lemmatize("winning"))
    # print(lem.lemmatize("winner"))
    i = 0
    j = 0
    k = 0
    m = 0
    n = 0
    name = ""
    host = ""
    cont = 0
    lis = []
    c = 1
    ex_cat = []
    estop = ['for', '...', 'award', 'at', "https", 'golden', 'http', '#', '.', '!', '?', '\\', ':', ';', '"', "'",
             'the', 'but', 'although', '#goldenglobes', 'and', '`', 'who', '&', "``"]
    if c == 1:

        # print(t['text'])
        for t in TWEETS:
            # for row in open('C:\\Users\\khaak\\Documents\\gg2015.txt', encoding="utf8").readlines():
            #     t = json.loads(row)
            # i=t['text'].find("host")
            # print(t["text"])
            i1 = re.search("best ", t, re.IGNORECASE)
            i2 = re.search("wins", t, re.IGNORECASE)
            i3 = re.search("bags", t, re.IGNORECASE)
            i4 = re.search("award", t, re.IGNORECASE)
            # i5 = re.search("award", t, re.IGNORECASE)
            l = t.find("best")
            k += 1
            host = ""
            count = 0
            val = 0
            if i1 != None:
                # doc = nlp(t['text'])
                # for ent in doc.ents:
                #     if(ent.label_=='PERSON'):
                #         print("the person identified")
                #         lis.append(ent.text)
                #
                #     print(ent.text, ent.start_char, ent.end_char, ent.label_)
                word_tokens = word_tokenize(t)
                # print("tokens",t["text"])
                count = 0
                append = 0
                categ = ""

                for i in word_tokens:
                    if i == "best" or i == "Best":
                        append = 1

                    if append == 1 and (i in estop):
                        # print("probab",t['text'])
                        if (count < 5):
                            cheeku = 1
                            # print("tweet less",t["text"])
                            # print(categ)
                        mul_categ.append(categ)
                        # print(t["text"])
                        # print("categ1", mul_categ[val])
                        val += 1
                        # print("categ",mul_categ)

                        categ = ""
                        count = 0
                        append = 0

                    if append == 1:
                        categ = categ + " " + i
                        count += 1

                    if append == 1 and count > 15:
                        categ = ""
                        count = 0
                        append = 0
                apeend = 0
                count = 0
                categ = ""
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
                n += 1
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
            if i4 != None:

                # print("cecil and award :",t['text'])
                doc = nlp(t)

                for ent in doc.ents:
                    # print("label",ent.label_,ent.text)
                    if "Golden" and "golden" not in ent.text and ("award" in ent.text or "Award" in ent.text):
                        # print("Award mila",ent.text)
                        st = ent.text
                        sp = st.lower().split(" ")
                        l = len(sp)
                        for j in range(0, l):
                            if sp[j] == 'award' and j == l - 1:
                                if j > 1:
                                    # print("Award mila", st,j)
                                    ex_cat.append(st)
                                break
                        # st=st[:it]
                        # print(st)
                        # ex_cat.append(st)
            #             st = ent.text
            #             it = st.find("Award")
            #             st = st[:it]
            #             print(st)
            #             ex_cat.append(st)

            if i2 != None or i3 != None:
                h = 0
                append = 0
                st = 0
                categ = ""
                count = 0
                val = 0
                word_tokens = word_tokenize(t)
                for i in word_tokens:
                    i = i.lower()
                    # print(i)
                    #############################################################################
                    if i == "wins" or i == "bags":
                        st = 1
                    if st == 1 and (i == "best" or i == "Best"):
                        append = 1
                    if append == 1 and (i in estop):
                        # print("wins waale", t['text'])
                        # print("categ",categ)
                        new_categ.append(categ)
                        # print(t["text"])
                        # print("categ1", mul_categ[val])
                        val += 1
                        # print("categ",mul_categ)
                        categ = ""
                        count = 0
                        append = 0

                    if append == 1:
                        categ = categ + " " + i
                        count += 1

                    if append == 1 and count > 15:
                        categ = ""
                        count = 0
                        append = 0
                apeend = 0
                count = 0
                categ = ""
                ###############################################################################################
                # print("congrats waale :",t['text'])
                # m+=1
    # print(lis)
    # print(lis)
    # print(Counter(lis))
    fil_categ = {}
    print("multiple_categ", mul_categ)
    print("new_categs", new_categ)

    t = collections.Counter(new_categ)
    g = dict(t)
    disc = []
    # print("val vhevlk", t)
    # print("val vhevlk", t[1])
    for r in t:
        # print("checking word ", r)
        c = 0
        l = 0
        for h in t:
            if r in h and r == h:
                c += 1
            elif r in h and r != h:
                l += 1
        # print("c and l", c, " ", l)
        if c == 1 and l == 0:
            # print("included:", r, t[r])
            fil_categ[r] = t[r]
        else:
            disc.append(r)

    # print("discared", disc)

    u = collections.Counter(fil_categ)
    ch = 0
    for i in u:
        ch += 1
    print(ch)
    h = collections.Counter(mul_categ)
    print("new categ1111", t)
    print("new categ222", t.most_common())
    print("new categ333", u.most_common(), len(u))
    print("new categ444", u.most_common(50), len(u))
    finalise = u.most_common(50)
    ans = []

    for hil in finalise:
        # print ("chek", hil[0])
        st = hil[0]
        if "," in hil[0]:
            st = st.replace(",", "-")
        st = st.strip()
        ans.append(st)
        # print("lat check", st, "chchhc", st[0:1])

    print(len(t))
    f = dict(h)
    # print("dict check",f[0])
    # print("multiple_categ 1",len(f))
    final_categ = {}
    # print("down",f)
    tot = 0

    filtered_awards = {}

    ex = collections.Counter(ex_cat)
    # print("etxra cat",Counter(ex_cat))
    for i in ex:
        if (ex[i] > 15):
            ans.append(i)
    print("final list is ", ans)
    for i in f.keys():
        for j in genre_lis:
            if j in i:
                filtered_awards[i] = f[i]
                break

    # print(filtered_awards)
    # print(len(filtered_awards))

    # for i in sorted(f.keys()):
    #     for j in f.keys():
    #         if i in j and j!=i:
    #             print(i,"and ",j)
    #             if i in final_categ:
    #                 del final_categ[i]
    #                 final_categ[j]=f[i]+f[j]
    #             else:
    #                 final_categ[j] = f[i] + f[j]
    #     # print("enteretd")
    # print("multiple_categ 2",(final_categ))
    # for i in sorted(f.keys()):
    #     if(f[i]>1):
    #         tot += 1
    #         print("categ no_",tot,":",i,h[i])
    # # print(Counter(mul_categ))
    print("total tweets", n, m)
    # print(ans)
    return ans


def get_nominees(year):
    '''Nominees is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change
    the name of this function or what it returns.'''
    global OFFICIAL_AWARDS_LIST, AWARD_TWEET_DICT, AWARD_NOMINEE_DICT, FILM_DATA, TWEETS, CLEAN_AWARD_NAMES
    # Your code here
    # TODO: Remove gloden globes stopwords
    ia = IMDb()

    stop_list_people = ['best', '-', 'award', 'for', 'or', 'made', 'in', 'a', 'by', 'performance', 'an', 'golden',
                        'globes', 'role', 'motion', 'picture', 'best', 'supporting']

    stoplist2 = ['golden globes', 'golden', 'globes', 'goldenglobes', '2020', 'best', 'motion', 'picture', 'best',
                 'supporting', '-', 'animated', 'best', 'comedy', 'drama', 'feature', 'film', 'foreign', 'globe',
                 'goes', 'golden', 'motion', 'movie', 'musical', 'or', 'original', 'picture', 'rt', 'series', 'song',
                 'television', 'to', 'tv']

    person_related_tweets = ['actor', 'actress', 'director', 'cecil']
    award_list = ['best motion picture - drama', 'best motion picture - comedy or musical', 'best television series - drama', 'best television series - comedy or musical', 'best animated feature film']
    words = ['didn\'t win', 'should\'ve won', 'should have won', 'did not win', 'deserved to win',
             'not win']
    firstWords = ['didn\'t', 'should\'ve', 'should', 'did', 'deserved', 'not']

    new_award_list = [a for a in OFFICIAL_AWARDS_LIST if a not in award_list]
    for award in new_award_list:
        print(award)
        if any(p in award for p in person_related_tweets):
            temp = []
            for tweet in AWARD_TWEET_DICT[award]:
                doc = nlp(tweet)
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        person = ent.text
                        if person not in temp:
                            temp.append(person)

            temp = merge_results(temp, "")
            ans = []
            for p in temp:
                person = ia.search_person(p)
                if person:
                    p1 = person[0]['name'].lower()
                    if p1 not in ans:
                        ans.append(p1)

            nominee_candidates = sorted(collections.Counter(ans).items(), key=lambda x: x[1], reverse=True)
            answer = [name[0] for name in nominee_candidates]

            if award == 'cecil b. demille award':
                AWARD_NOMINEE_DICT[award.lower()] = answer[:1]
            else:
                AWARD_NOMINEE_DICT[award.lower()] = answer[:10]

        elif award not in award_list:
            mandate = CLEAN_AWARD_NAMES[award][0]
            opt = CLEAN_AWARD_NAMES[award][1]

            ngrm_arr = []
            for tweet in AWARD_TWEET_DICT[award]:
                if all(word in tweet.lower() for word in mandate) and any(word in tweet.lower() for word in opt):
                    doc = nlp(tweet)
                    for ent in doc.ents:
                        # print(ent.text, ent.label_)
                        if ent.label_ == 'WORK_OF_ART' or ent.label_ == 'ORG' and all([s not in ent.text.lower() for s in stoplist2]):
                            movie = ent.text.replace('"', '')
                            ngrm_arr.append(movie)
            # print(ngrm_arr)
            ngrm_arr.sort(key=len, reverse=True)
            nominee_candidates = sorted(collections.Counter(ngrm_arr).items(), key=lambda x: x[1], reverse=True)
            answer = [name[0] for name in nominee_candidates]

            AWARD_NOMINEE_DICT[award] = answer[:10]

    mandate = CLEAN_AWARD_NAMES[award][0]
    movieType = 'movie'
    if 'television series' in award:
        movieType = 'tvSeries'
    woa = []
    search_list = ['motion picture', 'film', 'movie', 'picture', 'series']
    not_search_list = ['golden', 'globes', 'best', 'hbo', '@', '&', '//']
    for tweet in TWEETS:
        if any(word in tweet.lower() for word in search_list) and 'rt' not in tweet.lower():
            doc = nlp(tweet)
            for ent in doc.ents:
                if ent.label_ == 'WORK_OF_ART' and all(
                        word not in ent.text.lower() for word in not_search_list):
                    if ent.text not in woa:
                        woa.append(ent.text.replace('"', ''))
    print("woa", woa)
    print(FILM_DATA)
    new_arr = []
    for a in woa:
        FILM_DATA['indices'] = FILM_DATA["primaryTitle"].str.find(a, 0)
        for i in range(FILM_DATA.shape[0]):
            if FILM_DATA.iat[i, 4] != -1 and (mandate[0].lower() in FILM_DATA.iat[i, 3].lower()) and (FILM_DATA.iat[i, 0] == 'movie'):
                new_arr.append((FILM_DATA.iat[i, 0], FILM_DATA.iat[i, 1], FILM_DATA.iat[i, 3]))
                print(FILM_DATA.iat[i, 0], FILM_DATA.iat[i, 1], FILM_DATA.iat[i, 2], FILM_DATA.iat[i, 3], FILM_DATA.iat[i, 4])
        print('*****')
    print("new", new_arr)

    for award in award_list:
        mandate = CLEAN_AWARD_NAMES[award][0]
        movieType = 'movie'
        if 'television series' in award:
            movieType = 'tvSeries'
        AWARD_NOMINEE_DICT[award] = []
        temp = []
        for n in new_arr:
            if any(m in n[2] for m in mandate) and (movieType in n[0]):
                temp.append(n[1])
        AWARD_NOMINEE_DICT[award] = temp[:10]

    return AWARD_NOMINEE_DICT

    # for award in OFFICIAL_AWARDS_LIST:
    #     ngrm_arr = []
    #     mandate = CLEAN_AWARD_NAMES[award][0]
    #     opt = CLEAN_AWARD_NAMES[award][1]
    #     count = {'bf': 0, 'won': 0, 'wins': 0}
    #
    #     for tweet in TWEETS:
    #         for word in movie:
    #             if word in tweet.lower() and 'rt' not in tweet.lower():
    #                 print(tweet)
    #                 doc = nlp(tweet)
    #                 for ent in doc.ents:
    #                     print(ent.text, ent.label_)
    #                 print('****')

    #     tweet_arr = tweet.lower().split()
    #     # tweet_arr = [word for word in tweet_arr if word not in stop_words]
    #
    #     for t in tweet_arr:
    #         if t == 'best':
    #             for i in range(tweet_arr.index(t) + 1, len(tweet_arr)):
    #                 if tweet_arr[i] == 'for':
    #                     count['bf'] += 1
    #
    #                     for j in range(i + 1, len(tweet_arr)):
    #                         ngrm_arr.append(" ".join(tweet_arr[i + 1:j + 1]))
    #
    #         if t == "won":
    #             i = tweet_arr.index("won")
    #             count['won'] += 1
    #
    #             for j in range(i - 1, -1, -1):
    #                 ngrm_arr.append(" ".join(tweet_arr[j:i]))
    #
    #         elif t == "wins":
    #             i = tweet_arr.index("wins")
    #             count['wins'] += 1
    #
    #             for j in range(i - 1, -1, -1):
    #                 ngrm_arr.append(" ".join(tweet_arr[j:i]))
    #
    # new_arr = sorted(collections.Counter(ngrm_arr).items(), key=lambda x: x[1], reverse=True)
    # print(count)
    # print(new_arr)

    # if 'nomin' in tweet.lower() and all(word in tweet.lower() for word in mandate) and any(word in tweet.lower() for word in opt):

    # print(award)

    # for award in OFFICIAL_AWARDS_LIST:
    #     print(award)
    #     AWARD_NOMINEE_DICT[award] = []
    #     temp = []
    #     count = 0
    #     if any(p in award for p in person_related_tweets):
    #         for tweet in AWARD_TWEET_DICT[award]:
    #             doc = nlp(tweet)
    #             for ent in doc.ents:
    #                 if ent.label_ == "PERSON":
    #                     person = ent.text
    #                     if person not in temp:
    #                         temp.append(person)
    #                     '''
    #                     person = ia.search_person(ent.text)
    #                     if person:
    #                         p1 = person[0]['name'].lower()
    #                         print(p1)
    #                         if p1 not in temp:
    #                             temp.append(p1)
    #                     '''
    #         temp.sort(key=len, reverse=True)
    #         temp = merge_results(temp, "")
    #         ans = []
    #         for p in temp:
    #             person = ia.search_person(p)
    #             if person:
    #                 p1 = person[0]['name'].lower()
    #                 if p1 not in temp:
    #                     ans.append(p1)
    #         nominee_candidates = sorted(collections.Counter(ans).items(), key=lambda x: x[1], reverse=True)
    #         answer = [name[0] for name in nominee_candidates]
    #         if award == 'cecil b. demille award' or award == 'carol burnett award':
    #             AWARD_NOMINEE_DICT[award.lower()] = answer[:1]
    #         else:
    #             AWARD_NOMINEE_DICT[award.lower()] = answer[:10]
    #     else:
    #         '''
    #         ia.search_movie() --> 'title', 'kind'=series, 'year'=2019
    #         '''
    #         ngrm_arr = []
    #         ngrm_2 = []
    #         ngrm_3 = []
    #
    #         for tweet in AWARD_TWEET_DICT[award]:
    '''
                if 'nominate' in tweet:
                    print(tweet)
                    print('****')
                    doc = nlp(tweet)
                    for ent in doc.ents:
                        print(ent.text, ent.label_)
                    for token in doc:
                        if token.pos_ == "PROPN":
                            print(token.text, token.pos_)
                    print('****')

    '''

    #         doc = nlp(tweet)
    #         for ent in doc.ents:
    #             print(tweet)
    #             print("****")
    #             print(ent.text, ent.label_)
    #             if ent.label_ == 'WORK_OF_ART' and all([s not in ent.text.lower() for s in stoplist2]):
    #                 movie = ent.text.replace('"', '')
    #                 ngrm_arr.append(movie)
    #                 # print("movie: {}".format(movie))
    #
    #         ngrm_arr.sort(key=len, reverse=True)
    #         ngrm_arr = merge_results(ngrm_arr, award)
    #         genre = CLEAN_AWARD_NAMES[award][0]
    #
    #         new_arr = []
    #         for a in ngrm_arr:
    #             print(a)
    #             FILM_DATA['indices'] = FILM_DATA["primaryTitle"].str.find(a, 0)
    #             for i in range(FILM_DATA.shape[0]):
    #                 if FILM_DATA.iat[i, 3] != -1 and any(g in FILM_DATA.iat[i, 2].lower() for g in genre):
    #                     new_arr.append(FILM_DATA.iat[i, 0])
    #
    #         new_arr = sorted(collections.Counter(new_arr).items(), key=lambda x: x[1], reverse=True)
    #         answer = [name[0] for name in new_arr]
    #         AWARD_NOMINEE_DICT[award.lower()] = answer[:10]
    #         # print("ngrms", ngrm_arr.most_common(1))
    #         # ngrm_2 = collections.Counter(ngrm_2)
    #         # print("ngrms2", ngrm_2)
    #         # print("ngrms3", ngrm_3)
    #
    # print(answer)

    '''
                        movie_list = ia.search_movie(movie)
                        for m in movie_list:
                            if 'year' in m and m['year'] == 2019:
                                ngrm_arr.append(m['title'])
    '''
    '''
                words = ['didn\'t win', 'should\'ve won', 'should have won', 'did not win', 'deserved to win',
                         'not win']
                firstWords = ['didn\'t', 'should\'ve', 'should', 'did', 'deserved', 'not']


                for word in words:
                    if word in tweet:
                        count += 1

                
                tweet_arr = tweet.lower().split()
                # tweet_arr = [word for word in tweet_arr if word not in stop_words]

                for word in words:
                    if word in tweet.lower():
                
                        doc = nlp(tweet)
                        for ent in doc.ents:
                            if ent.label_ == "ORG":
                                ngrm_2.append(ent.text)
                '''
    '''
                    if t == 'drama':
                        i = tweet_arr.index(t)
                        for j in range(i + 1, len(tweet_arr)):
                            ngrm_arr.append(" ".join(tweet_arr[i + 1:j+1]))

                    '''
    '''
                    if t == 'best':
                        # flag = True
                        for i in range(tweet_arr.index(t) + 1, len(tweet_arr)):
                            if tweet_arr[i] == 'for':
                                count['bf'] += 1

                                for j in range(i + 1, len(tweet_arr)):
                                    ngrm_arr.append(" ".join(tweet_arr[i + 1:j + 1]))

                if not flag:
                    for t in tweet_arr:
                        if t == "won":
                            i = tweet_arr.index("won")
                            count['won'] += 1

                            for j in range(i - 1, -1, -1):
                                ngrm_arr.append(" ".join(tweet_arr[j:i]))

                        elif t == "wins":
                            i = tweet_arr.index("wins")
                            count['wins'] += 1

                            for j in range(i - 1, -1, -1):
                                ngrm_arr.append(" ".join(tweet_arr[j:i]))

                        '''

    '''
            bigrams_list = []
            for tweet in AWARD_TWEET_DICT[award]:
                print(tweet)
                print("***")

                doc = nlp(tweet)
                for token in doc:
                    if token.pos_ == "PROPN":
                        print(token.text)
                print("***")


                tweet = tweet.lower().split()
                tweet = [word for word in tweet if word not in stop_words]
                bigram = list(nltk.bigrams(tweet))
                bigrams_list.append(bigram)

            freq = nltk.FreqDist([b[0] + " " + b[1] for bigm in bigrams_list for b in bigm])
            # imdb search
            nominee_candidates = sorted(freq.items(), key=lambda x: x[1], reverse=True)
            '''

    '''
        ia = IMDb()
        for nom in nominee_candidates:
            res = ia.search_movie(nom[0])
            print(nom)
            print(res)
            print("----")
        '''

    # else:
    # AWARD_NOMINEE_DICT[award].append(nominee_candidates[:5])

    # print("{} nominees processed..".format(award))

    # print("AWARD: {}, COUNT: {}".format(award, count))
    # print ("nominee", AWARD_NOMINEE_DICT)
    # return AWARD_NOMINEE_DICT


def get_winner(year):
    '''Winners is a dictionary with the hard coded award
    names as keys, and each entry containing a single string.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    global CLEAN_AWARD_NAMES, OFFICIAL_AWARDS_LIST
    AWARD_WINNER_DICT = {}
    t = []
    mul_categ = []
    new_categ = []

    person = ["actor", "actress", "director", "cecil"]
    # stanford_ner_tagger = StanfordNERTagger(
    #     'stanford_ner/' + 'classifiers/english.muc.7class.distsim.crf.ser.gz',
    #     'stanford_ner/' + 'stanford-ner-3.9.2.jar'
    # )

    lem = WordNetLemmatizer()
    genre_lis = ["director", "actor", "actress", "award", "screenplay", "motion", "picture", "tv", "television",
                 "series", "drama", "comedy,musical,mini,series,cecil,song,score"]
    # print(lem.lemmatize("won"))
    # print(lem.lemmatize("winning"))
    # print(lem.lemmatize("winner"))
    i = 0
    j = 0
    k = 0
    m = 0
    n = 0
    name = ""
    host = ""
    cont = 0
    lis = []
    winner = {}
    ex_cat = []
    c = 1
    estop = ['for', '...', 'award', 'globe', 'at', "https", 'golden', 'http', '|', '#', '.', '!', '?', '\\', ':', ';',
             '"', "'", 'the', 'but', 'although', '#goldenglobes', 'and', '`', 'who', '&', "``"]
    if c == 1:
        # print(t['text'])
        for t in TWEETS:
            # for row in open('C:\\Users\\khaak\\Documents\\gg2015.txt', encoding="utf8").readlines():
            #     t = json.loads(row)
            # i=t['text'].find("host")
            # print(t["text"])
            i1 = re.search("best ", t, re.IGNORECASE)
            i2 = re.search("wins", t, re.IGNORECASE)
            i3 = re.search("bags", t, re.IGNORECASE)
            i4 = re.search("award", t, re.IGNORECASE)
            i5 = re.search("RT", t)
            l = t.find("best")
            k += 1
            host = ""
            count = 0
            val = 0
            # # if i1!=None :
            #     # doc = nlp(t['text'])
            #     # for ent in doc.ents:
            #     #     if(ent.label_=='PERSON'):
            #     #         print("the person identified")
            #     #         lis.append(ent.text)
            #     #
            #     #     print(ent.text, ent.start_char, ent.end_char, ent.label_)
            #     word_tokens = word_tokenize(t['text'])
            #     # print("tokens",t["text"])
            #     count=0
            #     append=0
            #     categ=""

            #     for i in word_tokens:
            #
            #         if i =="best" or i=="Best":
            #             append =1
            #
            #         if append==1 and (i in estop) :
            #             # print("stop word",i)
            #             if(count<5):
            #                 cheeku=1
            #                 # print("tweet less",t["text"])
            #                 # print(categ)
            #             mul_categ.append(categ)
            #             # print(t["text"])
            #             # print("categ1", mul_categ[val])
            #             val += 1
            #             # print("categ",mul_categ)
            #
            #             categ=""
            #             count=0
            #             append=0
            #
            #         if append == 1:
            #             # print("non stop",i)
            #             categ=categ+" "+i
            #             count+=1
            #
            #         if append==1 and count >15 :
            #             categ = ""
            #             count = 0
            #             append = 0
            #     append=0
            #     count=0
            #     categ=""
            #     # print("multiple_categ",mul_categ)
            #
            #     # print("CATEGO_1",t["text"])
            # #     for pos in nltk.pos_tag(word_tokens):
            # #         count+=1
            # #         if pos[1] =="NNP":
            # #             name=name+pos[0]
            # #             cont=1
            # #         elif cont==1 :
            # #             cont=0
            # #             host=host+name+"||||"
            # #             name=''
            # #         if re.search("host",pos[0],re.IGNORECASE):
            # #             if pos[1][0:2] == "VB":
            # #                 print("cnount now",count)
            # #                 print("breaking now")
            # #                 break
            # #     print("count",count)
            # #     # print("HOSTTTTTTTTTTTTTTTTTT",host)
            # #
            # #
            # #     # print("sea",t['text'])
            #     n+=1
            # #
            # # if i!=-1:
            # #     # print("============================================================================")
            # #     # print("Host:",t['text'])
            # #     # print("============================================================================")
            # #     # word_tokens = word_tokenize(t['text'])
            # #     # print("POS",nltk.pos_tag(word_tokens))
            # #     j+=1
            # #     # results = stanford_ner_tagger.tag(word_tokens)
            # #     # print("NER", nltk.ne_chunk(nltk.pos_tag(word_tokens)))

            if i5 == None and i1 != None and (i2 != None or i3 != None):
                h = 0
                # print("hi I was here")
                append = 0
                st = 0
                categ = ""
                count = 0
                val = 0
                word_tokens = word_tokenize(t)
                pe = ""
                mo = ""
                lis = []
                a = set()
                for i in word_tokens:
                    i = i.lower()
                    # print(i)
                    #############################################################################
                    if i == "wins" or i == "bags":
                        st = 1
                        doc = nlp(t)
                        # print(t["text"])
                        p = 0
                        m = 0
                        for ent in doc.ents:
                            if (ent.label_ == 'PERSON') and p == 0:
                                pe = ent.text
                                p = 1
                            if (ent.label_ == 'WORK_OF_ART') and m == 0:
                                mo = ent.text
                                m = 1
                            if p == 1 and m == 1:
                                break

                            # print(ent.label_," ",ent.text)
                            # if (ent.label_ == 'PERSON'):
                            #     # print("the person identified")
                            #     lis.append(ent.text)

                    if st == 1 and (i == "best" or i == "Best"):
                        append = 1
                    if append == 1 and (i in estop):
                        # print("wins waale", t)
                        # print("stop word",i)
                        new_categ.append(categ)
                        # print("categ",categ)
                        halwa = 0

                        for cat in person:
                            if cat in categ:
                                halwa += 1
                        if halwa >= 1:
                            if len(pe) > 3:
                                if categ not in winner:
                                    lis.append(pe)

                                    winner[categ] = lis
                                else:
                                    lis = winner[categ]
                                    lis.append(pe)
                                    winner[categ] = lis

                        else:
                            if len(mo) > 3:
                                if categ not in winner:
                                    lis.append(mo)
                                    winner[categ] = lis
                                else:
                                    lis = winner[categ]
                                    lis.append(mo)
                                    winner[categ] = lis
                        # print("new",new_categ)
                        # print(t["text"])
                        # print("categ1", mul_categ[val])
                        val += 1
                        # print("categ",mul_categ)
                        categ = ""
                        count = 0
                        append = 0

                    if append == 1:
                        # print("non stop",i)
                        categ = categ + " " + i
                        count += 1

                    if append == 1 and count > 20:
                        categ = ""
                        count = 0
                        append = 0
                append = 0
                count = 0
                categ = ""
                ###############################################################################################
                # print("congrats waale :",t['text'])
                # m+=1
    # print(lis)
    # print(lis)
    # print(Counter(lis))
    print(winner)
    fin = {}
    for y in winner.keys():
        lo = 0
        for ch in genre_lis:
            if ch in y:
                lo += 1
            if lo >= 1:
                fin[y] = winner[y]
    print("final", fin)

    for award in OFFICIAL_AWARDS_LIST:
        mandate = CLEAN_AWARD_NAMES[award][0]
        opt = CLEAN_AWARD_NAMES[award][1]
        temp = []
        for category in fin:
            # print(award, category)
            # print(mandate, opt)
            if all(word in category for word in mandate) and any(word in category for word in opt):
                temp.extend(fin[category])

        temp2 = collections.Counter(temp)
        print("temp", temp2)
        if len(temp2) > 0:
            AWARD_WINNER_DICT[award] = temp2.most_common(1)[0][0]
        else:
            AWARD_WINNER_DICT[award] = []

    # tup = collections.Counter(new_categ)
    # g=dict(t)
    # disc=[]
    # print("val vhevlk",t)
    # print("val vhevlk", tup)

    return AWARD_WINNER_DICT


def get_presenters(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    # Your code here
    single_presenter_pattern = re.compile(r'[A-Z][a-z]+\s[A-Z][a-z]+(?=\spresent)')
    multiple_presenters_pattern = re.compile(
        r'[A-Z][a-z]+\s[A-Z][a-z]+\sand\s[A-Z][a-z]+\s[A-Z][a-z]+(?=\spresent|\sintroduc)')
    stop = ["movie", "foreign", "golden", "award", "goldenglobes", "globes", "goldenglobes", "film"]

    ia = IMDb()
    AWARD_PRESENTER_DICT = {}
    for award in OFFICIAL_AWARDS_LIST:
        AWARD_PRESENTER_DICT[award] = []

        for tweet in AWARD_TWEET_DICT[award]:
            multiple_presenters = re.findall(multiple_presenters_pattern, tweet)

            for presenter in multiple_presenters:
                pp = presenter.split(' and ')
                p1 = pp[0]
                if any(word in p1 for word in stop):
                    continue

                pt = pp[1]
                ptt = pt.split(' ')
                p2 = " ".join(ptt[:2])
                if any(word in p2 for word in stop):
                    continue

                person = ia.search_person(p1)
                if person:
                    p1 = person[0]['name'].lower()
                person = ia.search_person(p2)
                if person:
                    p2 = person[0]['name'].lower()
                if p1 not in AWARD_PRESENTER_DICT[award]:
                    AWARD_PRESENTER_DICT[award].append(p1)
                if p2 not in AWARD_PRESENTER_DICT[award]:
                    AWARD_PRESENTER_DICT[award].append(p2)

            single_presenter = re.findall(single_presenter_pattern, tweet)
            for presenter in single_presenter:
                if any(word in presenter for word in stop):
                    continue
                person = ia.search_person(presenter)
                if person:
                    presenter = person[0]['name'].lower()
                if presenter not in AWARD_PRESENTER_DICT[award]:
                    AWARD_PRESENTER_DICT[award].append(presenter)

    print(AWARD_PRESENTER_DICT)
    return AWARD_PRESENTER_DICT


def pre_ceremony(YEAR):
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    global OFFICIAL_AWARDS_LIST, FILM_DATA
    '''
    if YEAR == 2013 or YEAR == 2015:
        f = open('award_names_1315.txt', 'r')
        awards_list = f.read()
        OFFICIAL_AWARDS_LIST = awards_list.split('\n')
    if YEAR == 2018 or YEAR == 2019 or YEAR == 2020:
        f = open('award_names_1920.txt', 'r')
        awards_list = f.read()
        OFFICIAL_AWARDS_LIST = awards_list.split('\n')
    '''
    # f = open('award_names_1315.txt', 'r')
    # awards_list = f.read()
    # OFFICIAL_AWARDS_LIST = ['best motion picture - drama']
    OFFICIAL_AWARDS_LIST = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress '
                                                           'in a motion picture - drama',
                            'best performance by an actor in a motion picture - drama', 'best motion picture - comedy '
                                                                                        'or musical',
                            'best performance by an actress in a motion picture - comedy or musical',
                            'best performance by an actor in a motion picture - comedy or musical', 'best animated '
                                                                                                    'feature film',
                            'best foreign language film', 'best performance by an actress in a supporting role in a '
                                                          'motion picture', 'best performance by an actor in a '
                                                                            'supporting role in a motion picture',
                            'best director - motion picture', 'best screenplay - motion picture', 'best original '
                                                                                                  'score - motion '
                                                                                                  'picture',
                            'best original song - motion picture', 'best television series - drama',
                            'best performance by an actress in a television series - drama', 'best performance by an '
                                                                                             'actor in a television '
                                                                                             'series - drama',
                            'best television series - comedy or musical', 'best performance by an actress in a '
                                                                          'television series - comedy or musical',
                            'best performance by an actor in a television series - comedy or musical',
                            'best mini-series or motion picture made for television', 'best performance by an actress '
                                                                                      'in a mini-series or motion '
                                                                                      'picture made for television',
                            'best performance by an actor in a mini-series or motion picture made for television',
                            'best performance by an actress in a supporting role in a series, mini-series or motion '
                            'picture made for television', 'best performance by an actor in a supporting role in a '
                                                           'series, mini-series or motion picture made for television']

    df = pd.read_csv("title.basics.tsv", sep="\t", usecols=['titleType', 'primaryTitle', 'startYear', 'genres'],
                     dtype={"titleType": object, "primaryTitle": object, "startYear": object, "genres": object})
    FILM_DATA = df.loc[(df['startYear'] == str(int(YEAR) - 1)) | (df['startYear'] == str(YEAR))]
    FILM_DATA.to_csv('film_data.csv')
    print("Size of filmDB: {}".format(FILM_DATA.shape))

    print("Pre-ceremony processing complete.")
    return


def main():
    '''This function calls your program. Typing "python gg_api.py"
    will run this function. Or, in the interpreter, import gg_api
    and then run gg_api.main(). This is the second thing the TA will
    run when grading. Do NOT change the name of this function or
    what it returns.'''
    global YEAR

    print("Enter YEAR")
    year = input()

    pre_ceremony()

    get_tweets()

    get_presenters(YEAR)
    get_nominees(YEAR)
    # explore()

    return


if __name__ == '__main__':
    main()
