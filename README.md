# twitterSentiment
Sentiment analysis of tweet dataset using the sentiment140 dataset from kaggle.
https://www.kaggle.com/kazanova/sentiment140

file structure required for the project:
```
project/  
  | input/  
  |   | dataset.csv  
  | controller.py  
  | evaluate.py  
  | predict.py  
  | read_data.py  
```

To run the code:
```python3 controller.py```

If you want to modify what parts of the code are run, edit ```controller.py```

This project was created for the cps803 class at Ryerson University. 
In it I use 3 models: LinearSVC, BernoulliNB and LogisticRegression
 to categorise tweets into 2 sentiment categories. I explore the different
 ways you can prepare data for these models and how well each of these preparations 
 performs. 

Warning: it takes a long time to run, make sure you start it before making dinner or something.
