from __future__ import division
import json
import re
import nltk
from nltk.corpus import stopwords

global TWEETS
TWEETS = {}

global AWARD_TWEET_DICTS
AWARD_TWEET_DICTS = {}

# golden globes stopwords
goldenGlobes_StopWords = ['golden', 'globes', 'goldenglobes', 'globe']
#Red Carpet Dress
best_dressed_keywords = []
worst_dressed_keywords = []
bestDressLt = []
worstDressLt = []

best_dressed_stopWords = ['best', 'dressed', 'also', 'red', 'carpet', 'redcarpet', 'looks', 'dress', 'best-dressed',
                   'worst-dressed', 'one', 'call', 'tonight', 'damn', 'men', 'women', 'love', 'beautifully', 'sexy',
                   'eonline', 'soon', 'thanks', 'obsessed', 'putting', 'list', 'thank']
best_dress_list = ['best', 'beautiful', 'pretty', 'love', 'sexy', 'beautifully','awesome','lovely','wow','gorgeous','wonderful','perfect']

worst_dressed_stopWords = ['best', 'worst', 'dressed', 'also', 'red', 'carpet', 'redcarpet', 'looks', 'dress', 'best-dressed',
                    'worst-dressed', 'one', 'call', 'tonight', 'damn', 'men', 'women', 'love', 'beautifully', 'sexy',
                    'eonline', 'soon', 'thanks', 'obsessed', 'putting', 'list', 'thank']
worst_dress_list = ['ugly', 'hate', 'awful', 'terrible', 'worst', 'sucks', 'not', 'isnt','bad','ugliest','yuck','hateful','laughable','controversial','dreadful','trashy']


OFFICIAL_AWARDS = ['cecil b. demille award', 'best motion picture - drama',
                   'best performance by an actress in a motion picture - drama',
                   'best performance by an actor in a motion picture - drama',
                   'best motion picture - comedy or musical',
                   'best performance by an actress in a motion picture - comedy or musical',
                   'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film',
                   'best foreign language film',
                   'best performance by an actress in a supporting role in a motion picture',
                   'best performance by an actor in a supporting role in a motion picture',
                   'best director - motion picture', 'best screenplay - motion picture',
                   'best original score - motion picture', 'best original song - motion picture',
                   'best television series - drama', 'best performance by an actress in a television series - drama',
                   'best performance by an actor in a television series - drama',
                   'best television series - comedy or musical',
                   'best performance by an actress in a television series - comedy or musical',
                   'best performance by an actor in a television series - comedy or musical',
                   'best mini-series or motion picture made for television',
                   'best performance by an actress in a mini-series or motion picture made for television',
                   'best performance by an actor in a mini-series or motion picture made for television',
                   'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television',
                   'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']


def pre_ceremony():
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''

    return


"""
def load_tweets(year):
    with open('gg2020.json',encoding="utf8") as f:
        data = [json.loads(line) for line in f]
        TWEETS[year] = [tweet['text'] for tweet in data]

    return TWEETS[year]

"""


# Loading Data
def load_tweets(year):
    if year in TWEETS:
        return TWEETS[year]
    else:
        try:

            if year == '2020':
                f = ('gg' + year + '.json')
                data = [json.loads(line) for line in open(f, 'r', encoding="utf8")]
                TWEETS[year] = [tweet['text'] for tweet in data]
                sort_tweets(year)
            else:
                f = open('gg' + year + '.json')
                data = json.load(f)
                # print("test")
                # print("data check",data)
                TWEETS[year] = [tweet['text'] for tweet in data]
                sort_tweets(year)
            return TWEETS[year]
        except Exception as e:
            print(e)
            return False


def sort_tweets(year):
    print("\nData Loading...".format(year))

    tweets = load_tweets(year)
    award_tweet_dict = {award: [] for award in OFFICIAL_AWARDS}

    stoplist = ['best', '-', 'award', 'for', 'or', 'made', 'in', 'a', 'by', 'performance', 'an', 'role']
    clean_award_names = {award: [[a for a in award.split() if not a in stoplist]] for award in OFFICIAL_AWARDS}

    substitutes = {}
    substitutes["television"] = ['tv', 't.v.']
    substitutes["motion picture"] = ["movie", "film"]
    substitutes["film"] = ["motion picture", "movie"]
    substitutes["comedy or musical"] = ['comedy', 'musical']
    substitutes["series, mini-series or motion picture made for television"] = ['series', 'mini-series', 'miniseries',
                                                                                'tv', 'television', 'tv movie',
                                                                                'tv series', 'television series']
    substitutes["mini-series or motion picture made for television"] = ['miniseries', 'mini-series', 'tv movie',
                                                                        'television movie']
    substitutes["television series"] = ['series', 'tv', 't.v.', 'television']
    substitutes["television series - comedy or musical"] = ["tv comedy", "tv musical", "comedy series", "t.v. comedy",
                                                            "t.v. musical", "television comedy", "television musical"]
    substitutes["television series - drama"] = ["tv drama", "drama series", "television drama", "t.v. drama"]
    substitutes["foreign language"] = ["foreign"]

    for award in OFFICIAL_AWARDS:
        for key in substitutes:
            if key in award:
                for sub in substitutes[key]:
                    alt_award = award.replace(key, sub)
                    clean_award_names[award].append([w for w in alt_award.split() if not w in stoplist])

    OFFICIAL_AWARDS.sort(key=lambda s: len(s), reverse=True)
    for award in OFFICIAL_AWARDS:
        # print "{} tweets unsorted".format(len(tweets))
        for i in range(len(tweets) - 1, -1, -1):
            tweet = tweets[i]
            for alt_award in clean_award_names[award]:
                contains_important_words = True
                for word in alt_award:
                    contains_important_words = contains_important_words and word.lower() in tweet.lower()

                if contains_important_words:
                    # print alt_award
                    award_tweet_dict[award].append(tweet)
                    del tweets[i]
                    break

    AWARD_TWEET_DICTS[year] = award_tweet_dict
    return


def get_award_tweet_dict(year):
    if year in AWARD_TWEET_DICTS:
        return AWARD_TWEET_DICTS[year]
    else:
        sort_tweets(year)
        return AWARD_TWEET_DICTS[year]

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
    if token in stopwords:
        return False

    tweet_stop = ['&amp;', 'rt', 'http']
    if '//t.co/' in token or token in tweet_stop:
        return False
#unicode
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

   # names= ['Kerry Washington','Nicole Kidman','Jessica','Salma Hayek','Jessica Chastain',' Halle Berry','sofia vargara','Getty','Anne Hathaway','Tina Fey','Daniel Craig','George Clooney','Ryan Seacrest','Jessica']
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

    return {
        "best_dress": list(best),
        "worst_dress": list(worst),
        "controversial_dress": list(controversial)
    }



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
                    worst_tokens = bigrams(nltk.word_tokenize(tweet), worst_dressed_keywords, goldenGlobes_StopWords + worst_dressed_stopWords)
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


def main():
    pre_ceremony()
    year = ""
    category = ""
    while year != "exit" and category != "exit":
        print('\nEnter year:  or "exit" to cancel.')
        print('--------------------------------')
        year = input()
        print('--------------------------------')

        found_tweets = True
        if year != "exit" and category != "different year":
            found_tweets = load_tweets(year)
            if found_tweets == False:
                print("\nNo data found for {}. Enter different year.".format(year))

        while year != "exit" and category != "different year" and category != "exit" and found_tweets:
            print(
                  'Enter "exit" to quit, or "different year" .')
            print('Enter 6. Dress on Red Carpet\n')
            print('--------------------------------')
            category = input()
            print('--------------------------------')

            if category == "6" or category.lower() == "most discussed on red carpet":
                print("\nThe most discussed people on the red carpet for {}...".format(year))
                # print(found_tweets)

                # RED CARPET DRESS
                dress_dict = redCarpet_dress(best_dressed(found_tweets), worst_dressed(found_tweets))
                print(f"Best dressed Celebrity at the {year} Golden Globes were: " + ', '.join(dress_dict['best_dress']))
                print(f"Worst dressed Celebrity were: " + ', '.join(dress_dict['worst_dress']))
                print(f"Most controversially dressed Celebrity were: " + ', '.join(dress_dict['controversial_dress']))



if __name__ == '__main__':
    main()
