'''Version 0.35'''
import json
import re
import collections
import nltk
from imdb import IMDb
import string
from nltk.corpus import stopwords
# from nltk.util import ngrams
import spacy

nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer

OFFICIAL_AWARDS_LIST = []
TWEETS = []
AWARD_TWEET_DICT = {}
YEAR = 2020
stop_words = set(stopwords.words('english'))


def get_tweets():
    global YEAR, TWEETS

    print("Loading tweets for year {}...".format(YEAR))
    f = 'gg{}.json'.format(YEAR)

    if YEAR == 2020:
        data = [json.loads(line) for line in open(f, 'r')]
        TWEETS = [t["text"].strip() for t in data]
        TWEETS = pre_process(TWEETS)
    else:
        fp = open(f, 'r')
        data = json.load(fp)

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


def generate_award_tweet_dict():
    global AWARD_TWEET_DICT, TWEETS, stop_words

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
    substitutes["Motion Picture"] = ["picture", "movie", "film", 'motion picture']
    substitutes["Television Series - Musical or Comedy"] = ['television series', 'tv series', "tv comedy", "tv musical",
                                                            "comedy series", "t.v. comedy",
                                                            "t.v. musical", "television comedy", "television musical"]
    substitutes["Musical or Comedy"] = ['musical', 'comedy']
    substitutes["Series, Limited Series or Motion Picture Made for Television"] = ['series', 'mini-series',
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
            if key in award:
                clean_award_names[award].append(substitutes[key])
                break
        if len(clean_award_names[award]) < 2:
            clean_award_names[award].insert(0, [])

    for award in OFFICIAL_AWARDS_LIST:
        AWARD_TWEET_DICT[award] = []
        for tweet in TWEETS:
            a = all([word in tweet.lower() for word in clean_award_names[award][0]])
            b = any([word in tweet.lower() for word in clean_award_names[award][1]])
            if a and b:
                AWARD_TWEET_DICT[award].append(tweet)

    AWARD_NOMINEE_DICT = {}

    # stoplist2 = stop_words.extend(['golden', 'globes'])
    stop_words.add('golden')
    stop_words.add('globes')
    stop_words.add('goldenglobes')
    person_related_tweets = ['Actor', 'Actress', 'Director', 'Cecil', 'Carol']
    for award in OFFICIAL_AWARDS_LIST:
        AWARD_NOMINEE_DICT[award] = []
        temp = []
        if any(p in award for p in person_related_tweets):
            for tweet in AWARD_TWEET_DICT[award]:
                doc = nlp(tweet)
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        temp.append(ent.text)
            nominee_candidates = sorted(collections.Counter(temp).items(), key=lambda x: x[1], reverse=True)
        else:
            '''
            ia.search_movie() --> 'title', 'kind'=series, 'year'=2019
            '''
            ngrm_arr = []
            for tweet in AWARD_TWEET_DICT[award]:
                # print(tweet)
                # print("***")
                flag = False
                tweet_arr = tweet.lower().split()
                tweet_arr = [word for word in tweet_arr if word not in stop_words]
                for t in tweet_arr:
                    if t == 'drama':
                        i = tweet_arr.index(t)
                        for j in range(i + 1, len(tweet_arr)):
                            ngrm_arr.append(" ".join(tweet_arr[i + 1:j+1]))
                    '''
                    if t == 'best':
                        flag = True
                        for i in range(tweet_arr.index(t)+1, len(tweet_arr)):
                            if tweet_arr[i] == 'for':
                                for j in range(i+1, len(tweet_arr)):
                                    ngrm_arr.append(" ".join(tweet_arr[i+1:j+1]))
                if not flag:
                    for t in tweet_arr:
                        if t == "won":
                            i = tweet_arr.index("won")
                            for j in range(i-1, -1, -1):
                                ngrm_arr.append(" ".join(tweet_arr[j:i]))
                        elif t == "wins":
                            i = tweet_arr.index("wins")
                            for j in range(i-1, -1, -1):
                                ngrm_arr.append(" ".join(tweet_arr[j:i]))
            '''
            print("ngrms", ngrm_arr)
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
        '''
        if award == 'Cecil B. DeMille Award' or award == 'Carol Burnett Award':
            AWARD_NOMINEE_DICT[award].append(nominee_candidates[0])
        else:
            AWARD_NOMINEE_DICT[award].append(nominee_candidates[:5])

        print("{} nominees processed..".format(award))
        '''
    # print(AWARD_NOMINEE_DICT)


def get_hosts(year):
    '''Hosts is a list of one or more strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here
    return hosts


def get_awards(year):
    '''Awards is a list of strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here
    return awards


def get_nominees(year):
    '''Nominees is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change
    the name of this function or what it returns.'''
    # Your code here
    # TODO: Remove gloden globes stopwords
    global AWARD_TWEET_DICT
    AWARD_NOMINEE_DICT = {}
    # if 'nominate' in tweet or 'nominated' in tweet:
    '''
    award = "best motion picture - drama"
    for i in range(len(AWARD_TWEET_DICT[award])):
        print(AWARD_TWEET_DICT[award][i])
        print("*****")
        doc = nlp(AWARD_TWEET_DICT[award][i])
        for ent in doc.ents:
            print("{} {}".format(ent.text, ent.label_))

    '''


def get_winner(year):
    '''Winners is a dictionary with the hard coded award
    names as keys, and each entry containing a single string.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    return winners


def get_presenters(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    # Your code here
    return presenters


def pre_ceremony():
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    # connect to IMDB
    global YEAR, OFFICIAL_AWARDS_LIST
    if YEAR == 2003 or YEAR == 2005:
        f = open('award_names_1315.txt', 'r')
        awards_list = f.read()
        OFFICIAL_AWARDS_LIST = awards_list.split('\n')
    if YEAR == 2018 or YEAR == 2019 or YEAR == 2020:
        f = open('award_names_1920_copy.txt', 'r')
        awards_list = f.read()
        OFFICIAL_AWARDS_LIST = awards_list.split('\n')

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
    YEAR = int(year)

    pre_ceremony()

    get_tweets()

    # get_nominees(YEAR)
    # explore()

    return


if __name__ == '__main__':
    main()
