import json
import re
import collections
import nltk
from imdb import IMDb
from nltk import word_tokenize
from fuzzywuzzy import fuzz
from textblob import TextBlob
import string
from nltk.corpus import stopwords
# from nltk.util import ngrams
import spacy
import pandas as pd
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from gender_detector.gender_detector import GenderDetector
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
AWARD_WINNER_DICT = {}
AWARD_PRESENTER_DICT = {}
HOSTS = []
FILM_DATA = pd.DataFrame()
YEAR = 2013
stop_words = set(stopwords.words('english'))
FUZZ_LIMIT = 90

OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']

# golden globes stopwords
goldenGlobes_StopWords = ['golden', 'globes', 'goldenglobes', 'globe']
# Red Carpet Dress
best_dressed_keywords = []
worst_dressed_keywords = []
bestDressLt = []
worstDressLt = []

best_dressed_stopWords = ['best', 'dressed', 'also', 'red', 'carpet', 'redcarpet', 'looks', 'dress', 'best-dressed',
                          'worst-dressed', 'one', 'call', 'tonight', 'damn', 'men', 'women', 'love', 'beautifully',
                          'sexy',
                          'eonline', 'soon', 'thanks', 'obsessed', 'putting', 'list', 'thank']
best_dress_list = ['best', 'beautiful', 'pretty', 'love', 'sexy', 'beautifully', 'awesome', 'lovely', 'wow', 'gorgeous',
                   'wonderful', 'perfect']

worst_dressed_stopWords = ['best', 'worst', 'dressed', 'also', 'red', 'carpet', 'redcarpet', 'looks', 'dress',
                           'best-dressed',
                           'worst-dressed', 'one', 'call', 'tonight', 'damn', 'men', 'women', 'love', 'beautifully',
                           'sexy',
                           'eonline', 'soon', 'thanks', 'obsessed', 'putting', 'list', 'thank']
worst_dress_list = ['ugly', 'hate', 'awful', 'terrible', 'worst', 'sucks', 'not', 'isnt', 'bad', 'ugliest', 'yuck',
                    'hateful', 'laughable', 'controversial', 'dreadful', 'trashy']


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

    generate_award_tweet_dict_old()
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
    global OFFICIAL_AWARDS_LIST, AWARD_TWEET_DICT

    stoplist = ['-', 'award', 'for', 'or', 'made', 'in', 'a', 'by', 'an', 'role', 'best']

    person_related_tweets = ['actor', 'actress', 'director', 'cecil']

    substitutes = {}
    substitutes["limited series or motion picture made for television"] = ['miniseries', 'mini-series', 'tv movie',
                                                                           'television movie', 'motion picture']
    substitutes["motion picture"] = ["picture", "movie", 'motion picture']
    substitutes["television series - musical or comedy"] = substitutes['television series - comedy or musical'] = [
        'television series', 'tv series', "tv comedy", "tv musical",
        "comedy series", "t.v. comedy",
        "t.v. musical", "television comedy", "television musical"]
    substitutes["musical or comedy"] = substitutes["comedy or musical"] = ['musical', 'comedy']
    substitutes["series, limited series or motion picture made for television"] = substitutes[
        'series, miniseries or motion picture made for television'] = ['series', 'mini-series',
                                                                       'miniseries', 'limited series',
                                                                       'tv', 'television', 'tv movie',
                                                                       'tv series', 'television series',
                                                                       'motion picture']
    substitutes["television series - drama"] = ['televisin series', 'tv series', "tv drama", "drama series",
                                                "television drama", "t.v. drama"]
    substitutes["television series"] = ['series', 'tv', 't.v.', 'television']
    substitutes["television"] = ['tv', 't.v.']

    threshold = 3
    for award in OFFICIAL_AWARDS_LIST:
        if len(award.split()) > 5:
            threshold = 4
        else:
            threshold = 3
        # print(award)
        line = ''
        for sub in substitutes:
            if sub in award:
                line = substitutes[sub]
                break
            # print(award)
            AWARD_TWEET_DICT[award] = []
            for tweet in TWEETS:
                count = 0
                if 'actor' in award and 'actor' not in tweet.lower():
                    continue
                if 'actress' in award and 'actress' not in tweet.lower():
                    continue
                if 'cecil' in award and 'cecil' not in tweet.lower():
                    continue
                if 'director' in award and 'director' not in tweet.lower():
                    continue
                if not any(p in award for p in person_related_tweets) and any(
                        p in tweet.lower() for p in person_related_tweets):
                    continue
                for cat in award.split(' '):
                    if line and any(s in tweet.lower() for s in substitutes[line]):
                        count += 1
                    if cat not in stoplist and cat in tweet.lower():
                        count += 1
                    if count == threshold:
                        AWARD_TWEET_DICT[award].append(tweet)
                        break

        # print(AWARD_TWEET_DICT[award][:10])
        # print(len(AWARD_TWEET_DICT[award]))


def generate_award_tweet_dict_old():
    global AWARD_TWEET_DICT, TWEETS, CLEAN_AWARD_NAMES, stop_words

    print("Processing Tweets...")

    award_tweet_dict = {award: [] for award in OFFICIAL_AWARDS_LIST}
    stoplist = ['best', '-', 'award', 'for', 'or', 'made', 'in', 'a', 'by', 'performance', 'an', 'role', 'motion',
                'picture', 'television', 'limited', 'series', 'musical', 'comedy']
    clean_award_names = {award: [[a for a in award.lower().split(' ') if not a in stoplist]] for award
                         in OFFICIAL_AWARDS_LIST}

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


def get_hosts(year):
    '''Hosts is a list of one or more strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here
    global HOSTS

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

            n += 1

        if i != -1:
            j += 1

        if i == -1 and l != -1:

            m += 1

    t = collections.Counter(lis)
    host_fil = []
    host_dic = {}
    disc = []
    t = t.most_common(7)

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

    ratio = (fin[1][1] / fin[0][1])
    ans = []
    ans.append(fin[0][0])
    if ratio > 0.65:
        ans.append(fin[1][0])

    # hosts = []
    HOSTS = ans
    return ans


def get_awards(year):
    '''Awards is a list of strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here

    t = []
    mul_categ = []
    new_categ = []


    lem = WordNetLemmatizer()
    genre_lis = ["director", "actor", "actress", "award", "screenplay", "motion", "picture", "tv", "television",
                 "series", "drama", "comedy,musical"]

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

        for t in TWEETS:

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

                n += 1

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

    fil_categ = {}
    # print("multiple_categ", mul_categ)
    # print("new_categs", new_categ)

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
    # print(ch)
    h = collections.Counter(mul_categ)
    # print("new categ1111", t)
    # print("new categ222", t.most_common())
    # print("new categ333", u.most_common(), len(u))
    # print("new categ444", u.most_common(50), len(u))
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

    # print(len(t))
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
    # print("final list is ", ans)
    for i in f.keys():
        for j in genre_lis:
            if j in i:
                filtered_awards[i] = f[i]
                break


    # print("total tweets", n, m)
    # print(ans)

    # ==> use this
    # results = getattr(gg_api, 'get_%s' % info_type)(year)
    return ans


def get_nominees(year):
    global OFFICIAL_AWARDS_LIST, AWARD_TWEET_DICT, AWARD_NOMINEE_DICT, FILM_DATA, TWEETS, CLEAN_AWARD_NAMES
    # Your code here
    # TODO: Remove gloden globes stopwords
    ia = IMDb()

    stop_list_people = ['best', '-', 'award', 'for', 'or', 'made', 'in', 'a', 'by', 'performance', 'an', 'golden',
                        'globes', 'role', 'motion', 'picture', 'best', 'supporting']

    stoplist2 = ['golden globes', 'golden', 'globes', 'goldenglobes', '2020', 'best', 'motion', 'picture', 'best',
                 'supporting', '-', 'animated', 'best', 'comedy', 'drama', 'feature', 'film', 'foreign', 'globe',
                 'goes', 'golden', 'motion', 'movie', 'musical', 'or', 'original', 'picture', 'rt', 'series',
                 'song',
                 'television', 'to', 'tv']

    person_related_tweets = ['actor', 'actress', 'director', 'cecil']
    award_list = ['best motion picture - drama', 'best motion picture - comedy or musical',
                  'best television series - drama', 'best television series - comedy or musical',
                  'best animated feature film']
    words = ['didn\'t win', 'should\'ve won', 'should have won', 'did not win', 'deserved to win',
             'not win']
    firstWords = ['didn\'t', 'should\'ve', 'should', 'did', 'deserved', 'not']

    new_award_list = [a for a in OFFICIAL_AWARDS_LIST if a not in award_list]
    for award in new_award_list:
        # print(award)
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
                        if ent.label_ == 'WORK_OF_ART' or ent.label_ == 'ORG' and all(
                                [s not in ent.text.lower() for s in stoplist2]):
                            movie = ent.text.replace('"', '')
                            ngrm_arr.append(movie)
            # print(ngrm_arr)
            ngrm_arr.sort(key=len, reverse=True)
            nominee_candidates = sorted(collections.Counter(ngrm_arr).items(), key=lambda x: x[1], reverse=True)
            answer = [name[0] for name in nominee_candidates]

            AWARD_NOMINEE_DICT[award] = answer[:10]

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
    # print("woa", woa)

    new_arr = []
    for a in woa:
        if a.title() in FILM_DATA.values:
            new_arr.append(a)

        # print("new", new_arr)
        for award in award_list:
            AWARD_NOMINEE_DICT[award] = new_arr[:10]


    return AWARD_NOMINEE_DICT


def get_winner(year):
    global TWEETS
    # t=[]
    # lem=WordNetLemmatizer()
    # print(lem.lemmatize("won"))
    # print(lem.lemmatize("winning"))
    # print(lem.lemmatize("winner"))
    wordfreq = {}
    co = 0
    title = ['for', "wins", "congrats", "i", '...', 'top', 'award', 'globes', 'globe', 'at', "https", 'golden', 'http',
             '|', '#', '.', '!', '?', '\\', ':', ';', '"', "'", 'the', 'but', 'although', '#goldenglobes', 'and', '`',
             'who', '&', "``"]
    estop = ['for', 'at', "https", 'golden' 'http', '#', '.', ',', '!', '-', '?', '\\', ':', ';', '"', "'", 'the',
             'but', 'although']
    OFFICIAL_AWARDS = ['cecil b. demille award', 'best motion picture - drama',
                       'best performance by an actress in a motion picture - drama',
                       'best performance by an actor in a motion picture - drama',
                       'best motion picture - comedy or musical',
                       'best performance by an actress in a motion picture - comedy or musical',
                       'best performance by an actor in a motion picture - comedy or musical',
                       'best animated feature film',
                       'best foreign language film',
                       'best performance by an actress in a supporting role in a motion picture',
                       'best performance by an actor in a supporting role in a motion picture',
                       'best director - motion picture', 'best screenplay - motion picture',
                       'best original score - motion picture', 'best original song - motion picture',
                       'best television series - drama',
                       'best performance by an actress in a television series - drama',
                       'best performance by an actor in a television series - drama',
                       'best television series - comedy or musical',
                       'best performance by an actress in a television series - comedy or musical',
                       'best performance by an actor in a television series - comedy or musical',
                       'best mini-series or motion picture made for television',
                       'best performance by an actress in a mini-series or motion picture made for television',
                       'best performance by an actor in a mini-series or motion picture made for television',
                       'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television',
                       'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']

    a = set()
    list = []
    pre = {}
    # print(t['text'])
    for t in TWEETS:

        co += 1
        # print(co)

        word_tokens = word_tokenize(t)

        for i in OFFICIAL_AWARDS_LIST:
            if i.lower() in t.lower():
                # print(i,"tweet-->",t["text"])
                new_list = []
                lis = []
                tit = ""
                for h in word_tokens:
                    if h.istitle() and h.lower() not in title and h.lower() not in i:
                        tit = tit + " " + h
                    else:
                        if tit != "":
                            new_list.append(tit.strip())
                            tit = ""

                if i not in pre:
                    pre[i] = new_list
                else:
                    lis = pre[i]
                    for h in lis:
                        new_list.append(h)
                    pre[i] = new_list


        sent = ""

    ans = {}
    for j in OFFICIAL_AWARDS:
        ci = 0
        for i in a:
            if i == j:
                ci += 1
        if ci == 0:
            # print("nf", j)
            res = get_winner_old([j])
            ans[j] = res[j]

    for i in pre:
        ans[i] = collections.Counter(pre[i]).most_common(1)[0][0]


    return ans


def get_winner_old(award_given):
    '''Winners is a dictionary with the hard coded award
    names as keys, and each entry containing a single string.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    global CLEAN_AWARD_NAMES, OFFICIAL_AWARDS_LIST, AWARD_WINNER_DICT
    AWARD_WINNER_DICT = {}
    t = []
    mul_categ = []
    new_categ = []

    person = ["actor", "actress", "director", "cecil"]


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
    title = ['for', "wins", "i", '...', 'top', 'award', 'globes', 'globe', 'at', "https", 'golden', 'http', '|', '#',
             '.', '!', '?', '\\', ':', ';', '"', "'", 'the', 'but', 'although', '#goldenglobes', 'and', '`', 'who', '&',
             "``"]
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
                tit = ""
                new_list = []
                a = set()
                for i in word_tokens:
                    if i.istitle() and i.lower() not in title and st == 0:
                        tit = tit + " " + i
                    else:
                        if tit != "":
                            new_list.append(tit.strip())
                            tit = ""
                    i = i.lower()
                    # print(i)
                    #############################################################################
                    if i == "wins" or i == "bags":
                        st = 1
                        doc = nlp(t)
                        # print(t["text"])
                        p = 0
                        m = 0

                    if st == 1 and (i == "best" or i == "Best"):
                        append = 1
                    if append == 1 and (i in estop):
                        # print("wins waale", t)
                        # print("stop word",i)
                        new_categ.append(categ)
                        if categ not in winner:
                            winner[categ] = new_list
                        else:
                            lis = winner[categ]
                            for h in lis:
                                new_list.append(h)
                            winner[categ] = new_list

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
    # print(winner)
    fin = {}
    for y in winner.keys():
        lo = 0
        for ch in genre_lis:
            if ch in y:
                lo += 1
            if lo >= 1:
                fin[y] = winner[y]
    answer = {}
    for m in fin:
        res = []
        t = collections.Counter(fin[m]).most_common(5)
        # print(m, ":", t)
        if len(t) > 0:
            for kl in range(len(t)):
                res.append(t[kl][0])
            answer[m] = res
    # print("final", fin)

    for award in award_given:
        mandate = CLEAN_AWARD_NAMES[award][0]
        opt = CLEAN_AWARD_NAMES[award][1]
        temp = []
        for category in answer:
            # print(award, category)
            # print(mandate, opt)
            if all(word in category for word in mandate) and any(word in category for word in opt):
                temp.extend(answer[category])

        temp2 = collections.Counter(temp)
        # print("temp", temp2)
        if len(temp2) > 0:
            arr = temp2.most_common(5)
            # print(arr)
            for tup in arr:
                if tup[arr.index(tup)][0]:
                    AWARD_WINNER_DICT[award] = tup[0]
                    break
        else:
            AWARD_WINNER_DICT[award] = []

    # tup = collections.Counter(new_categ)
    # g=dict(t)
    # disc=[]
    # print("val vhevlk",t)
    # print("val vhevlk", tup)

    return AWARD_WINNER_DICT


def generate_json(YEAR):
    global HOSTS, AWARD_WINNER_DICT, AWARD_NOMINEE_DICT, AWARD_PRESENTER_DICT, OFFICIAL_AWARDS_LIST
    answer = {}
    answer['hosts'] = HOSTS

    for award in OFFICIAL_AWARDS_LIST:
        temp = {}
        temp["Presenters"] = AWARD_PRESENTER_DICT[award]
        temp["Nominees"] = AWARD_NOMINEE_DICT[award]
        temp["Winner"] = AWARD_WINNER_DICT[award]
        answer[award] = temp

    name = "g{}answers.json".format(YEAR)
    with open(name, 'w') as f:
        json.dump(answer, f)


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
    global AWARD_PRESENTER_DICT
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

    # print(AWARD_PRESENTER_DICT)
    return AWARD_PRESENTER_DICT


def pre_ceremony(YEAR):
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    global OFFICIAL_AWARDS_LIST, FILM_DATA

    if YEAR == 2013 or YEAR == 2015:
        OFFICIAL_AWARDS_LIST = OFFICIAL_AWARDS_1315
    else:
        OFFICIAL_AWARDS_LIST = OFFICIAL_AWARDS_1819

    # f = open('award_names_1315.txt', 'r')
    # awards_list = f.read()
    # OFFICIAL_AWARDS_LIST = ['best motion picture - drama']

    df = pd.read_csv("film_data.csv", usecols=['titleType', 'primaryTitle', 'startYear', 'genres'],
                     dtype={"titleType": object, "primaryTitle": object, "startYear": object, "genres": object})
    FILM_DATA = df.loc[(df['startYear'] == str(int(YEAR) - 1)) | (df['startYear'] == str(YEAR))]
    # FILM_DATA.to_csv('film_data.csv')
    print("Size of filmDB: {}".format(FILM_DATA.shape))

    print("Pre-ceremony processing complete.")
    return


def main():
    '''This function calls your program. Typing "python gg_api.py"
    will run this function. Or, in the interpreter, import gg_api
    and then run gg_api.main(). This is the second thing the TA will
    run when grading. Do NOT change the name of this function or
    what it returns.'''
    global YEAR, TWEETS

    print("Enter YEAR")
    year = input()

    pre_ceremony(YEAR)
    get_tweets(YEAR)
    get_hosts(YEAR)
    get_awards(YEAR)
    get_nominees(YEAR)
    get_presenters(YEAR)
    get_winner(YEAR)
    generate_json(YEAR)
    print("Running Additional Tasks")
    hashtag_trends(YEAR)
    sentiment(YEAR)
    bd = best_dressed(TWEETS)
    wd = worst_dressed(TWEETS)
    redCarpet_dress(bd, wd)

    return


def hashtag_trends(YEAR):
    t = []
    mul_categ = []
    new_categ = []

    lem = WordNetLemmatizer()
    genre_lis = ["director", "actor", "actress", "award", "screenplay", "motion", "picture", "tv", "television",
                 "series", "drama", "comedy,musical"]

    i = 0
    j = 0
    k = 0
    m = 0
    n = 0
    name = ""
    host = ""
    cont = 0
    lis = []
    has = []
    estop = ['for', 'at', "https", 'golden', 'http', '#', '.', '!', '-', '?', '\\', ':', ';', '"', "'", 'the', 'but',
             'although', '#goldenglobes', 'and', '`', 'who', '&']

    f = 'gg{}.json'.format(YEAR)

    if YEAR == 2020:
        data = [json.loads(line) for line in open(f, 'r')]
    else:
        fp = open(f, 'r')
        data = json.load(fp)

    for t in data:
        # print("hey",t["timestamp_ms"])
        # ts = t["timestamp_ms"] / 1000
        s = t["text"].split(" ")
        for i in s:
            if '#' in i:
                i = i.lower()
                if "golden" not in i[1:]:
                    has.append(i)
                    # print(i[1:])

    ed = collections.Counter(has).most_common(30)
    x = []
    y = []
    for i in ed:
        # print(i[0])
        x.append(i[0])
        y.append(i[1])



    plt.bar(x, y)
    # plt.yticks(y, x)

    plt.xticks(rotation=90)
    plt.ylabel('Frequency')
    plt.xlabel('Hashtags')
    plt.title('Hashtags Trend')
    plt.show()

    # print("total tweets", n, m)


def sentiment(YEAR):

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
    estop = ['for', 'at', "https", 'golden', 'http', '#', '.', '!', '-', '?', '\\', ':', ';', '"', "'", 'the', 'but',
             'although', '#goldenglobes', 'and', '`', 'who', '&']

    time = []
    y = []
    count = 0

    f = 'gg{}.json'.format(YEAR)

    if YEAR == 2020:
        data = [json.loads(line) for line in open(f, 'r')]
    else:
        fp = open(f, 'r')
        data = json.load(fp)

    for t in data:
        # print("hey",t["timestamp_ms"])
        ts = t["timestamp_ms"]
        # dt_object = datetime.fromtimestamp(ts)
        time.append(ts)
        # print("dt_object =", dt_object)
        # print("type(dt_object) =", type(dt_object))
        str = t["text"].replace("Best", "")
        blob = TextBlob(str)
        y.append(blob.sentences[0].sentiment.polarity)
        count += 1
        # print(count)
        # print(y)
    int = time[0]
    count = 0
    sum1 = 0
    sum2 = 0
    x = 0
    xax = []
    yax = []
    for i in range(0, len(time)):
        diff = time[i] - int
        # print(diff)
        # print("count", count)
        if float(diff) > 10000:
            # print("change", diff, " ", count)
            x = x + 10000
            y1 = (sum2 / count)
            xax.append(x)
            yax.append(y1)
            int = time[i]
            count = 1
            sum1 = 0
            sum2 = 0
        else:
            sum1 = sum1 + time[i]
            sum2 = sum2 + y[i]
            count += 1
    l = len(y)
    # print(xax)
    # print(yax)
    plt.plot(xax, yax)
    plt.ylabel('Sentiment')
    plt.xlabel('Time')
    plt.title('Sentiment Trend')
    plt.show()

    # print("total tweets", n, m)


def bigrams(tokens, valid_keyword, invalid_keyword):
    flag = False
    bigrams = set()
    for token in tokens:
        if valid_token(token, valid_keyword, invalid_keyword):
            if flag:
                bigrams.add(flag + ' ' + token)
            flag = token
    return bigrams


def valid_token(token, valid_keyword, invalid_keyword):
    token = token.lower()
    if token in valid_keyword:
        return True
    if token in invalid_keyword:
        return False
    if token in stopwords.words():
        return False

    tweet_stop = ['&amp;', 'rt', 'http']
    if '//t.co/' in token or token in tweet_stop:
        return False
    # unicode
    if any(ord(c) > 128 for c in token):
        return False

    regex = re.compile('[^a-zA-Z]')
    token = regex.sub('', token)
    if len(token) < 1:
        return False
    return True


def redCarpet_dress(best_dress, worst_dress):
    Sch_amount = 10
    best_dress = [i[0] for i in best_dress if i[1] > Sch_amount]
    worst_dress = [i[0] for i in worst_dress if i[1] > Sch_amount]
    best = set()
    worst = set()
    controversial = set()

    # names = ['Kerry Washington', 'Nicole Kidman', 'Jessica', 'Salma Hayek', 'Jessica Chastain', ' Halle Berry',
    #          'sofia vargara', 'Getty', 'Anne Hathaway', 'Tina Fey', 'Daniel Craig', 'George Clooney', 'Ryan Seacrest',
    #          'Jessica']
    names = set(line.strip() for line in open('names.txt'))

    for celeb_1 in best_dress:
        if len(best) == 2 and len(controversial) == 2:
            break

        if celeb_1 in names:
            if celeb_1 in best_dress and celeb_1 in worst_dress:
                if len(controversial) < 2:
                    controversial.add(celeb_1)
            else:
                if len(best) < 2:
                    best.add(celeb_1)
    for celeb_2 in worst_dress:
        if len(worst) == 2 and len(controversial) == 2:
            break

        if celeb_2 in names:
            if celeb_2 in best_dress and celeb_2 in worst_dress:
                if len(controversial) < 2:
                    controversial.add(celeb_2)
            elif celeb_2 in worst_dress:
                if len(worst) < 2:
                    worst.add(celeb_2)

    f = open("red_carpet_results.txt", "w")
    f.write("best_dress")
    f.write('\n')
    for b in best:
        f.write(b)
        f.write('\n')
    f.write("worst_dress")
    f.write('\n')
    for w in worst:
        f.write(w)
        f.write('\n')
    f.write("controversial_dress")
    f.write('\n')
    for c in controversial:
        f.write(c)
        f.write('\n')
    f.close()



def best_dressed(tweets):
    best_dressed_dictionary = {}
    for tweet in tweets:
        if 'dress' in tweet:
            if any(i in tweet for i in worst_dress_list):
                continue
            else:
                if any(i in tweet for i in best_dress_list):
                    best_tokens = bigrams(nltk.word_tokenize(tweet), best_dressed_keywords,
                                          goldenGlobes_StopWords + best_dressed_stopWords + best_dress_list)
                    # print('tokens',tokens)
                    for tokens in best_tokens:
                        tokens = tokens.lower()
                        if tokens not in best_dressed_dictionary:
                            best_dressed_dictionary[tokens] = 1
                        else:
                            best_dressed_dictionary[tokens] += 1
    best_dress_lt = sorted(best_dressed_dictionary.items(), key=lambda x: x[1], reverse=True)

    return best_dress_lt


def worst_dressed(tweets):
    worst_dressed_dictionary = {}
    for tweet in tweets:
        if 'dress' in tweet:
            for stop_Words in worst_dress_list:
                if stop_Words in tweet:
                    worst_tokens = bigrams(nltk.word_tokenize(tweet), worst_dressed_keywords,
                                           goldenGlobes_StopWords + worst_dressed_stopWords)
                    for tokens in worst_tokens:
                        tokens = tokens.lower()
                        if tokens not in worst_dressed_dictionary:
                            worst_dressed_dictionary[tokens] = 1
                        else:
                            worst_dressed_dictionary[tokens] += 1
    worst_dressed_lt = sorted(worst_dressed_dictionary.items(), key=lambda x: x[1], reverse=True)
    bestDress = best_dressed(tweets)
    for bd in bestDress:
        if bd[1] > 35:
            bestDressLt.append(bd[0])
    for wd in worst_dressed_lt:
        if wd[0] in bestDressLt:
            pass
        else:
            worstDressLt.append(wd)
    return worstDressLt


if __name__ == '__main__':
    main()