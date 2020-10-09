# SentimentAnalysisPythonApp

## Overview
The purpsose of the sentiment-analysis-mean-app is to provide a simple sentiment analysis mean app. The application accepts text in the form of a 'Post' and forwards that text to a python tensorflow application. The application gets the text through pub/sub between itself and the server. The predicted sentiment is then returned and displayed to the web app. 

## Inference
This folder is where the main app for sentiment analysis lives. Follow the README in the `inference` directory to get the application up and running and hooked into the sentiment analysis mean app. 

## Train
This folder is not required for the sentiment analysis mean app. However, It is good to have around fo tweaking, recreating, and learnign purposes. Follow the README in the `train` directory to get started.

