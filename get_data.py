import pandas as pd

df = pd.read_csv("title.basics.tsv", sep="\t", usecols=['titleType', 'primaryTitle', 'startYear', 'genres'],
                     dtype={"titleType": object, "primaryTitle": object, "startYear": object, "genres": object})
FILM_DATA = df.loc[(df['startYear'] == '2012') | (df['startYear'] == '2013') | (df['startYear'] == '2014') | (df['startYear'] == '2015')| (df['startYear'] == '2016') | (df['startYear'] == '2017') | (df['startYear'] == '2018')| (df['startYear'] == '2019')| (df['startYear'] == '2020')]
FILM_DATA.to_csv('film_data.csv')
print("Size of filmDB: {}".format(FILM_DATA.shape))