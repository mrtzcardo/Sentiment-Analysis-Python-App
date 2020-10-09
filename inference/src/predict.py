import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import re
import sys

import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_text

import redis
import time

def sent(predictedSentiment):
    if np.argmax(predictedSentiment) == 0:
        return "Negative Sentiment"
    else:
        return "Positive Sentiment"
        

def main():

    use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

    model = keras.models.load_model('my_model.h5')

    r = redis.Redis(host='sentiment-analysis-broker', port=6379)
    p = r.pubsub()

    p.subscribe('sentiment-request')

    print("loaded models")
    while True:
        message = p.get_message()
        if message:
            print("recieved data")
            emb_input = use([str(message['data'])])
            review_emb_input = tf.reshape(emb_input, [-1]).numpy()
            predictedSentiment = model.predict(emb_input)
            r.publish('sentiment-reply', sent(predictedSentiment))
        time.sleep(0.001)


if __name__ == "__main__":
    main()
