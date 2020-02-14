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

1. Clone repo or download zip, then navigate to directory
2. Create a virtual environment if needed and activate the environment
3. Run $ pip install -r  requirements.txt
4. Put "ggYYYY.json", "ggYYYYanswers.json" and "autograder.py" files in the root directory.
6. Run gg_apifake.py followed by autograder.py separately for each year.
  (gg_apifake.py would need a year to run and creates a json named --> answersYYYY.json
  run autograder to get the scores --> (gg_api uses the above json to generate the answers for the autograder)
 7. Additional Task (defined inside gg_apifake.py) are : 
 a) Trending Hashtags (saves image as "hashtag-trends.png") 
 b) Tweet Sentiments with respet to time (saves image as "sentiments.png")
 c) Worst Dressed d) Best Dressed e) Controversial Dress (saves as a .txt file named "red_carpet_results.txt") 
   
    
