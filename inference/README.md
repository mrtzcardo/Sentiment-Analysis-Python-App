# SentimentAnalysisPythonApp

## Overview
The purpsose of the sentiment-analysis-python-app is to provide a simple sentiment analysis inferencing app. The application subcribes to a redis server that receives text data from the sentiment-analysis-mean-app and preforms sentiment analysis on the text. The predicted sentiment is then publishedback to the redis server for the web app. 

## Install Dependencies
Install docker
sentiment-analysis-mean-app

## Build
### Network 
Run `docker network create -d bridge sentiment-analysis-bridge`

### Dev
From this directory run: `docker build -t sentiment-analysis-python:dev -f docker/Dockerfile.dev src`
### Prod
From this directory run: `docker build -t sentiment-analysis-python:prod -f docker/Dockerfile.prod src`

## Start redis

This app depends on a redis server running on the same bridge network with the resovable hostname `sentiment-analysis-broker`. The easiest way to get redis up and runnign is with their Docker image.
`docker run --network=sentiment-analysis-bridge --name=sentiment-analysis-broker --rm redis:alpine`

## Development server
### Dev
Run `docker run --network=sentiment-analysis-bridge --rm sentiment-analysis-python:dev`
### Prod
Run `docker run --network=sentiment-analysis-bridge --rm sentiment-analysis-python:prod`