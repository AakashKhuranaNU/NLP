Team members: Aakash Khurana, Unnati Parekh, Iram Naaz


Packages:
* fuzzywuzzy
* IMDbPY
* nltk
* spacy
* pandas
* numpy
* matplotlib
* textblob


How to run:
*NOTE: Output json file (named 'answers.json') will be created when autograder.py is run. 


1. Clone repo or download zip, then navigate to directory
2. Create a virtual environment if needed and activate the environment
3. Run $ pip install -r  requirements.txt
4. Put "ggYYYY.json" files in the root directory, such as gg2013.json or gg2015.json 
5.To get autograder scores run gg_apifake.py followed by autograder.py
(gg_apifake.py would need a year to run and creates a json named --> ggYYYYanswers.json
 run autograder to get the scores --> (gg_api uses the above json to generate the answers for the autograder)
 6. Additional Task are : a) Trending Hashtags b) Tweet Sentiments with respet to time c) Worst Dressed d) Best Dressed e) Controversial Dress
    - Additional tasks are defined inside gg_api.py
    
