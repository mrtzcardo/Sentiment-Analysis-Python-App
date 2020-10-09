# SentimentAnalysisTrain

## Overview
This application is used to train a network to preform sentiment analysis on any length of text. This [Blog](https://www.curiousily.com/posts/sentiment-analysis-with-tensorflow-2-and-keras-using-python/) was followed and adopted to create this training app. 

TRAINING IS NOT NECCESSARY FOR SENTIMENT ANALYSIS APP. Just for fun/tweaking


## Dependencies
- Download training data from [kaggle](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe) into the src directory. 

To retrain the model go to the server/training folder. The steps we taken from this 
Dependencies for training: `numpy pandas tensorflow sklearn tensorflow_hub tensorflow_text tqdm pyyaml h5py`

## Build

From the docker directory run: `docker build -t sentiment-analysis-trianer:dev .`

## Train

From the src directory `docker run --mount type=bind,source="$(pwd)",destination=/src sentiment-analysis-trianer:dev`