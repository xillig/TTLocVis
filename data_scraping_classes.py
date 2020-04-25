# 1. Data Scraping: WriteFileListener class

# Contents of the class "WriteFileListener"
# - Streaming of tweets via the Twitter API.
# - Results saved as JSON.

# Additional information:
# To be able to use Twitter data, one first needs to have access to it. Therefore, a Twitter Developer Account is
# necessary to access the Twitter API. This account can be obtained through an application process on the Twitter website,
# in which one has to enter his or her personal data and a description of the project one plans to use the Twitter data with.
# In general, Twitter needs to ensure that its users' data is not used for activities that are harmful towards its
# own users or third parties. Particularly sensitive are processes in which political discrimination could occur
# through the creation of a profile of individual users by the developer account holder. For example,
# this concerns user data on his / her sexuality, religion or trade union membership (Twitter 2019a).
# Apart from that, Twitter data is public information that users knowingly shared with the world and thus is
# not limited in its use.


from datetime import datetime
from http.client import IncompleteRead
import json
import numpy as np
import os
import pandas as pd
from queue import Queue
import sys
import time
import tweepy as tw
from tweepy import OAuthHandler
from tweepy import API
from tweepy.streaming import StreamListener
import urllib3.exceptions
from urllib3.exceptions import ProtocolError
from urllib3.exceptions import ReadTimeoutError


# Attention!
# It's advised to keep the StreamListener object as sparse as possible to reduce computing time to prevent an
# "Incomplete Read" error. This error occurs if the API provides more data than can be processed.
# The occurring "data traffic jam" leads the program to throw an exception. The incoming data in "on_data" is
# being queued and then computed over several threads to reduce computing time even more. Also save the streamed
# tweets locally, writing via a usb-connection to an external drive for example, takes to much time and causes
# an "incomplete read" error!

# Set up the StreamListener object. For more information,
# see: https://tweepy.readthedocs.io/en/latest/streaming_how_to.html

# __Details about the following block:__
# The most important method is on_data. It is used to access every single tweet in raw form. As a result,
# operations can now be performed directly on the incoming raw data. In this case, these are very simple
# actions: A tweet, which is transmitted in the form of a JSON string, is queried to include the substring
# “extended_tweet”, as well as the substring "#". This is a meta-filtering performed directly on the raw data:
# Only tweets that contain more than 140 characters have this root-level attribute called "extended_tweets".
# Likewise, only tweets that contain at least one hashtag include the substring “#”.
# Only tweets that satisfy these two If-conditions (as well as other filters defined through the Stream Object,
# see below) are stored in the empty JSON file.
# Another method of the class handles the rate limit. When occurring, it interrupts the streaming when the rate limit
# has been reached. If the API object is set to wait_on_rate_limit = True, the method will not be called.
# The on_error method handles all other errors that might occur while using the Streaming API (Twitter 2019d).
# The error code and the time are returned, the stream ends.
# Finally, the on_timeout method deals with the loss of the host's Internet connection and tries to reconnect to
# the server after a certain time. Thus, streaming is automatically resumed in the case of short-term internet
# disconnections.

# The Streaming API of Twitter is addressed by passing the incoming stream (tweets in JSON format) through an
# instantiated stream object. A default Tweepy-provided python class called StreamListener inherits methods
# from the self-defined WriteFileListener class. Modifications in WriteFileListener are provided to specify
# any filter conditions regarding collected tweets. WriteFileListener can then instantiate a user-specific
# listener object, which is passed to a stream object (instantiated from the StreamListener class) to start
# a session using the authentication information of the Twitter developer account to access the raw data (Tweepy 2019).
# When a listener object from the class WriteFileListener is instantiated, the object opens an empty JSON
# file in the folder where the script “Data Scraping” itself is stored. Its name contains the respective
# time of instantiation, in order to refer the streaming start time to the respective file.

# Streaming: run the StreamListener object, collect tweets and handle errors
# (assure streaming continues if errors occur)

# sources: Pfaffenberger (2016): Twitter als Basis wissenschaftlicher Studien: Eine Bewertung
# gängiger Erhebungs- und Analysemethoden der Twitter-Forschung
# https://stackoverflow.com/questions/48034725/tweepy-connection-broken-incompleteread-best-way-to-handle-exception-or-can
# https://github.com/tweepy/tweepy/issues/908

# set up the StreamListener object. For information see: http://docs.tweepy.org/en/v3.4.0/streaming_how_to.html

class WriteFileListener(StreamListener):
    # __Details about the following block:__
    # Necessary information include the so-called consumer key and consumer secret, which identify the personal access of the account for the Twitter API, as well as the access token and access token secret, which do regulate the access of the application to the API (it is necessary to have an App in the Twitter developer account to access tweets). Tweepy reads this data into an OAuthHandler object that governs the communication between the Tweepy functions and the Twitter API. There are many ways to configure this wrapper function. In this case, special attention should be paid on handling the Rate Limit. Twitter specifies a rate limit to prevent a user or bot from abusing the API by making too many queries simultaneously or within a short period of time. In case the rate limits are breached, a (temporary) ban of the IP address will be imposed, meaning that one can no longer query data (Twitter 2019c).
    # The API-wrapper function handles this automatically if the argument wait_on_rate_limit = True is passed: The query process is stopped when the rate limit is reached until the IP address is released again, thus preventing it from being blocked.

    # Authentification procedure: After getting an Twitter developer account, one has to verifiy him-/herself via the personal key, token and secrets. Tweepy is using this information to access the Twitter API.

    def __init__(self, auth_path, languages, locations, save_path=os.getcwd(), extended=True, hashtag=True):
        super(StreamListener, self).__init__()
        self.auth_path = auth_path  # path containing the Twitter API Information
        self.api = self.access_auth()
        self.extended = extended
        self.hashtag = hashtag
        self.languages = languages
        self.locations = locations
        now = datetime.now()
        self.save_file = open(os.path.join(save_path, 'tweets ' + now.strftime('%Y%m%d-%H%M%S') + '.json'), 'w')
        self.streaming()

    def access_auth(self):
        # authentication procedure:
        with open(self.auth_path, 'r') as twitter_access:
            twitter_access = list(twitter_access)

        # insert your own consumer and access details here!!!
        consumer_key = twitter_access[0].rstrip()
        consumer_secret = twitter_access[1].rstrip()
        access_token = twitter_access[2].rstrip()
        access_secret = twitter_access[3].rstrip()

        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_secret)
        api = API(auth, wait_on_rate_limit=True,
                  # Streaming: API waits if rate limit is reached and gives out an notification
                  wait_on_rate_limit_notify=True)
        return api

    def stream_process(self):
        self.stream = tw.Stream(auth=self.api.auth, listener=self, tweet_mode='extended')
        now = datetime.now()
        print('Start streaming: ' + now.strftime('%Y%m%d-%H%M%S'))
        self.stream.filter(languages=self.languages, locations=self.locations)

    def streaming(self):
        while True:
            try:
                self.stream_process()

            except KeyboardInterrupt:  # exit via Ctrl+D or Kernel -> Interrupt
                now = datetime.now()
                print("Stopped at: " + now.strftime('%Y%m%d-%H%M%S'))
                self.stream.disconnect()
                return False

            except ProtocolError:  # catches the "data traffic jam" error: Incomplete Read: https://stackoverflow.com/questions/48034725/tweepy-connection-broken-incompleteread-best-way-to-handle-exception-or-can
                now = datetime.now()
                print(
                    'Incomplete read error - too many tweets were posted live at the same time at your location!' + now.strftime(
                        '%Y%m%d-%H%M%S'))
                self.stream.disconnect()
                time.sleep(60)
                continue

            except ReadTimeoutError:  # catches, if my internet connection is lost
                self.stream.disconnect()
                print(now.strftime('%Y%m%d-%H%M%S') + ': ReadTimeoutError exception! Check your internet connection!')
                return False
        return

    def on_data(self, tweet):
        if 'extended_tweet' in tweet and self.extended == True and '#' in tweet and self.hashtag == True:  # save only tweets with more than 140 characters and hashtag
            self.save_file.write(str(tweet))
        elif 'extended_tweet' in tweet and self.extended == True and self.hashtag == False:
            self.save_file.write(str(tweet))
        elif self.extended == False and '#' in tweet and self.hashtag == True:
            self.save_file.write(str(tweet))
        elif self.extended == False and self.hashtag == False:
            self.save_file.write(str(tweet))

    # disconnects the stream if API rate limit is hit:
    def on_limit(self, status_code):
        if status_code == 420:
            now = datetime.now()
            print('API Rate limit reached: ' + now.strftime('%Y%m%d-%H%M%S'))
            return False

    # catches all errors that are delivered by the Twitter API:
    def on_error(self, status_code):
        now = datetime.now()
        print(now.strftime('%Y%m%d-%H%M%S') + ' Error: ' + str(status_code))  # Print Error-Status code.
        return False

    # catches, it internet connection gets lost:
    def on_timeout(self):
        print('Timeout: Wait 120 sec.')
        time.sleep(120)  # timeout 120 sec. if connection is lost.
        return
