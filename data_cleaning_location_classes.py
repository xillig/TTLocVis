# # 2. Tweepy Data cleaning
##
# # Contents of the script:
# - Data cleaning: removing duplicates, removing quoted tweets and retweets.
# - Text cleaning: removal of Hyperlink-embeddings and mentions (usernames), identify and handle hashtags and Emojis.
# - Handling the location data: access bounding box coordinates and calculate its center.
# - Accessing user meta-data for every tweet.
# - Removing unnecessary data. 
# - Results saved as pkl.

# # Overview about all covarates that are availiable after running that script:
# - created_at - timestamp of the creation of the corresponding tweet.
# - extended_tweet - shows the complete text of a tweet if it is longer than 140 characters. Else None.
# - id - the tweets id as integer. 
# - id_str - the tweets id as string.
# - place - sub-dictionary: contains information about the tweets associated location.
# - source - hyperlink to the Twitter website, where the tweet object is stored.
# - text - shows the complete text of a tweet, regardless of whether it’s longer than 140 characters or not.
# - user - sub-dictionary: contains information about the tweets’ associated user.
# - emojis - contains the emoji(s) of a tweet.
# - hashtags - contains the hashtag(s) of a tweet (without “#”)
# - bounding_box.coordinates_str - contains all bounding box coordinates as a string. Originates from place.
# - center_coord_X - the X-coordinate of the center of the bounding box.
# - center_coord_Y - the Y-coordinate of the center of the bounding box.
# - retweet_count - number of retweets of the corresponding tweet.
# - favorite_count - number of favorites of the corresponding tweet.
# - user_created_at - timestamp of the users’ profile creation. Originates from user.
# - user_description -  textual description of users’ profile. Originates from user.
# - user_favourites_count - The total number of favorites for all of the users tweets. Originates from user.
# - user_followers_count - The total number of followers of the user. Originates from user.
# - user_friends_count - The total number of users followed by the user. Originates from user.
# - user_id - profile id of the users profile as integer. Originates from user.
# - user_listed_count - The number of public lists which this user is a member of. Originates from user.
# - user_location - self-defined location by the user for the profile. Originates from user.
# - user_name - self-defined name for the user themselfs. Originates from user.
# - user_screen_name - alias of the self-defined name for the user themselfs. Originates from user.
# - user_statuses_count - number of tweets published by the user (incl. retweets). Originates from user.
# 


import ast
from datetime import datetime
from functools import partial
import gensim
import gensim.corpora as corpora
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.test.utils import datapath
import glob
from heapq import nlargest
from importlib import reload
import itertools as it
import json
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing import cpu_count
import numpy as np
import os
import pandas as pd
from pandas.io.json import json_normalize
import pickle
import random as rm
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import tweepy as tw
from tweepy import OAuthHandler
from tweepy import API
import warnings


# # Read-in the Raw data:

# __Details about the following blocks:__ 
# At the beginning, all JSON files were read-in and merged together. This was necessary to ensure that only complete JSON strings were read. While streaming, it can sometimes happen that the stream stops during the saving process of a tweet or that an error occurs. In that case, an incomplete JSON string would be saved, which would lead to an error message. The script catches this error when reading-in the JSON files by checking the code for each tweet, provided in JSON-string format, on whether the tweets string is complete or not. If it is not, the incomplete string is ignored and the next one is read-in.
# This is achieved with a try-except combination, in which the try block tries to read-in the JSON string by json.loads(), a function that decodes a JSON string. If this function returns a ValueError, the next iteration of the loop over all JSON strings calls the except block located in the try-except construct. In the following, the JSON file, which is assembled by individual JSON strings, is converted into a pandas DataFrame named raw_data and the data is given a first table structure.

# Combine all raw tweet-objects in JSON format (all JSONs that are in the specified folder):

# Read-in the raw data:
# Problem: the stream collects data in one file, therefore many different json-objects have to be read in.
# The tweets json-strings are separeted while beeing read in.

# source: https://stackoverflow.com/questions/20400818/python-trying-to-deserialize-multiple-json-objects-in-a-file-with-each-object-s

class Cleaner(object):

    def __init__(self, load_path, data_save_name='my_cleaned_and_tokenized_data', languages=None, metadata=False,
                 min_tweet_len=None, spacy_model='en_core_web_sm'):
        self.data_save_name = data_save_name
        self.languages = languages
        self.load_path = load_path
        self.metadata = metadata
        self.min_tweet_len = min_tweet_len
        self.spacy_model = spacy.load(spacy_model)  # loading the statistical spacy-model
        self.raw_data = self.loading()
        self.raw_data = self.cleaning()

    def loading(self):
        json_data = []
        for filename in glob.glob(os.path.join(self.load_path, '*.json')):
            try:
                with open(filename, 'r') as f:
                    for line in f:
                        while True:
                            try:
                                # check if a json-object is complete ( "}" will occur to close the "{", json.loads(.)
                                # won't throw an error. )
                                jfile = json.loads(line)  # "jfile" is a sanity check.
                                break
                            except ValueError:
                                # Not yet a complete JSON value
                                line += next(f)

                        # do something with jfile
                        json_data.append(json.loads(line))
            except:
                next

        if len(json_data) > 150000:  # set a maximum value to prevent "json.dumps"-function from crashing.
            print('Please read-in a maximum of 150.000 tweets per object!')
            return False

        df_data = pd.read_json(json.dumps(json_data))  # turn the list back to a json and then into a pd.dataframe!

        return df_data

    def cleaning(self):
        # delete duplicates, check for US origin and language and polygon bounding_box-geotag:
        self.raw_data = self.raw_data.drop_duplicates('id')  # remove duplicates
        self.raw_data = self.raw_data[self.raw_data['is_quote_status'] == False]  # remove quoted statuses
        self.raw_data = self.raw_data[self.raw_data['retweeted'] == False]  # remove retweets
        if self.languages is not None:
            self.raw_data = self.raw_data[self.raw_data['lang'] == self.languages]  # check for language
        # getting the indexes, to check sub-json 'raw_data['place']':
        self.raw_data['place'] = self.raw_data['place'].fillna('')  # handling the "None"-values
        self.raw_data = self.raw_data[self.raw_data['place'] != '']  # take only tweets with bounding_box-geodata
        place_df = json_normalize(self.raw_data['place'])  # read the geo-location sub-json in as data frame
        poly_indices = place_df.index[place_df['bounding_box.type'] == 'Polygon'].to_numpy()  # check is location is
        # available and turn indices object to numpy array.

        # get a sub-df with the conditions above met:
        self.raw_data = self.raw_data.iloc[poly_indices, :]
        place_df = place_df.iloc[poly_indices, :]

        # if tweet is longer than 140 chars: The extended tweet-text must be submitted to the 'text' column:
        indices_extw = np.array(self.raw_data[self.raw_data['extended_tweet'].notna()].index.tolist())  # get indices of
        # extended tweets.
        ex_tweet_df = self.raw_data['extended_tweet']  # get the extended tweets sub-json
        ex_tweet_df = json_normalize(ex_tweet_df[indices_extw])  # normalize it
        ex_tweet_df = ex_tweet_df['full_text']  # save the full text as list
        ex_tweet_df = pd.Series(ex_tweet_df)
        ex_tweet_df = pd.Series(ex_tweet_df.values,
                                index=indices_extw)  # change the list to a Series and put the right indices on
        self.raw_data.loc[indices_extw, 'text'] = ex_tweet_df[indices_extw]  # overwrite the data in 'text',
        # where the tweet is extended.

        # split the string at the occurrence of the embedded hyperlink and take the first part over all
        # entries (remove hyperlinks):
        self.raw_data['text'] = self.raw_data['text'].apply(lambda x: re.split('https://t.co', x)[0])

        # remove and append Emojis:
        if self.metadata == True:
            emojis = re.compile(
                u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')  # emoji unicode
            indices = np.array(self.raw_data['text'].index.tolist())  # save the index numbers of the entries
            l1 = []
            for i in self.raw_data['text']:
                l1.append(emojis.findall(i))
            l1 = pd.Series(l1)
            l1 = pd.Series(l1.values, index=indices)  # put the gathered values together with the old indices
            l1 = l1.rename('emojis')
            self.raw_data = pd.concat([self.raw_data, l1], axis=1)  # concat the series with the emojis to our dataframe

        self.raw_data['text'] = self.raw_data['text'].apply(
            lambda x: x.encode('ascii', 'ignore').decode('ascii'))  # remove emojis from textfield
        # remove mentions (usernames):
        self.raw_data['text'] = self.raw_data['text'].apply(
            lambda x: ' '.join(i for i in x.split() if not i.startswith('@')))

        # collect hashtags from the text:
        lists = self.raw_data['text'].str.split()  # split every word in the text at whitespace
        indices = np.array(lists.index.tolist())  # save the index numbers of the entries
        # make a new list and collect all hashtag-words:
        l1 = []
        for i in lists:
            l2 = []
            for j in i:
                if j.startswith('#'):
                    a = re.split('[^#a-zA-Z0-9-]', j)  # remove all non-alphanumeric characters at end of hashtag
                    l2.append(a[0])
            l1.append(l2)

        l1 = pd.Series(l1)
        l1 = pd.Series(l1.values, index=indices)  # put the gathered values together with the old indices
        l1 = l1.rename('hashtags')
        self.raw_data = pd.concat([self.raw_data, l1], axis=1)  # concat the series to our dataframe
        self.raw_data['text'] = self.raw_data['text'].str.replace('#', '')

        if self.min_tweet_len is not None:
            # check the length of a tweet:
            len_text = self.raw_data['text'].apply(lambda x: len(x))  # get the length of all text fields
            self.raw_data = self.raw_data[
                len_text > self.min_tweet_len]  # take only texts with more than 100 characters

        # append the location data:
        place_df = json_normalize(self.raw_data['place'])  # update 'place_df' to remaining numbers of tweets
        indices = np.array(self.raw_data.index.tolist())  # update indices
        st = place_df['bounding_box.coordinates'].apply(lambda x: str(x))  # convert all entries to strings
        st = pd.Series(st)  # list to series
        st = pd.Series(st.values, index=indices)  # insert updated indices
        st = st.str.replace('[', '')  # remove all unneeded symbols
        st = st.str.replace(']', '')
        st = st.apply(lambda x: re.split(',', x))  # split the string to isolate every number
        st = pd.DataFrame(st)
        st = st.rename(columns={0: "bounding_box.coordinates_str"})  # rename the column

        # Calculate the center of the bounding box:
        # LONG FIRST; LAT LATER: center of rectangle for first entry: x1=st[0][1], x2=st[0][3], y1=st[0][0], y2=st[0][4]
        # xy-center: (x1+x2)/2, (y1+y2)/2
        st['val1'] = st['bounding_box.coordinates_str'].apply(
            lambda x: float(x[1]))  # append the needed values as new column
        st['val3'] = st['bounding_box.coordinates_str'].apply(lambda x: float(x[3]))  # and convert them to float
        st['val0'] = st['bounding_box.coordinates_str'].apply(lambda x: float(x[0]))
        st['val4'] = st['bounding_box.coordinates_str'].apply(lambda x: float(x[4]))

        st['center_coord_X'] = (st['val1'] + st['val3']) / 2  # bounding box-center x-coordinate
        st['center_coord_Y'] = (st['val0'] + st['val4']) / 2  # bounding box-center y-coordinate
        self.raw_data = pd.concat([self.raw_data, st], axis=1)  # append the X and Y coordinates to our dataframe

        # Predicts part-of-speech tags, dependency labels, named entities and more.
        # Tokenization:
        self.raw_data['text_tokens'] = self.raw_data['text'].apply(lambda x: Cleaner._tokenizer(self.spacy_model, x))

        if self.metadata == True:
            user_df = json_normalize(self.raw_data['user'])  # unpack the nested dict
            # pick interesting columns
            user_df = user_df.loc[:, ['created_at', 'description', 'favourites_count', 'followers_count',
                                      'friends_count', 'id', 'listed_count', 'location', 'name', 'screen_name',
                                      'statuses_count']]
            # rename interesting columns
            user_df = user_df.rename(columns={'created_at': 'user_created_at', 'description': 'user_description',
                                              'favourites_count': 'user_favourites_count',
                                              'followers_count': 'user_followers_count',
                                              'friends_count': 'user_friends_count', 'id': 'user_id',
                                              'listed_count': 'user_listed_count',
                                              'location': 'user_location', 'name': 'user_name',
                                              'screen_name': 'user_screen_name',
                                              'statuses_count': 'user_statuses_count'})
            user_df.index = self.raw_data.index  # transfer the indices from main df to sub-df
            self.raw_data = self.raw_data.join(user_df)  # join the interesting columns to the main df

            # remove unnecessary columns:
            self.raw_data = self.raw_data.drop(
                ['contributors', 'coordinates', 'display_text_range', 'entities', 'extended_entities',
                 'favorited', 'filter_level', 'geo', 'in_reply_to_screen_name', 'in_reply_to_status_id',
                 'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str',
                 'is_quote_status',
                 'lang', 'possibly_sensitive', 'quote_count', 'quoted_status', 'quoted_status_id',
                 'quoted_status_id_str',
                 'quoted_status_permalink', 'reply_count', 'retweeted', 'truncated', 'val1', 'val3',
                 'val0', 'val4', 'timestamp_ms'], axis=1)

        else:
            self.raw_data = self.raw_data.loc[:, ['created_at', 'text', 'text_tokens', 'hashtags', 'center_coord_X',
                                                  'center_coord_Y']]

        return self.raw_data

    @staticmethod  # using static method, see: https://realpython.com/instance-class-and-static-methods-demystified/
    def _tokenizer(nlp, text):
        #  "nlp" Object is used to create documents with linguistic annotations.
        doc = nlp(text)

        # Create list of word tokens
        token_list_general = []
        token_list_topic_model = []
        for token in doc:
            if (token.is_stop == False) & (token.is_alpha == True) & (token.pos_ != 'PUNCT') & (len(token) > 2) & (
                    re.search('amp', str(token)) == None) & ((token.pos_ == 'PROPN') | (token.pos_ == 'NOUN')
                                                             | (token.pos_ == 'ADJ') | (token.pos_ == 'VERB')):
                token_list_topic_model.append(token.lemma_.lower())  # tokens for topic models

        return token_list_topic_model

    # _stat_func_caller = _tokenizer.__func__(spacy_model,b) #ensure callability of static method inside instance method, see:
    # https://stackoverflow.com/questions/12718187/calling-class-staticmethod-within-the-class-body

    def saving(self, save_path, type='pkl'):

        _pack_size = 300000  # no of tweets saved in one go
        parts_to_save = math.ceil(len(self.raw_data) / _pack_size)  # calculate how many parts to save (rounds up)
        upper_bound = _pack_size
        for i in range(0, parts_to_save):
            lower_bound = upper_bound - _pack_size
            file_to_save = self.raw_data.iloc[lower_bound:upper_bound, :]
            upper_bound = upper_bound + _pack_size

            if type == 'pkl':
                file_to_save.to_pickle(os.path.join(save_path, self.data_save_name + '_part_' + str(i + 1) + '.pkl'))
            else:
                file_to_save.to_csv(os.path.join(save_path, self.data_save_name + '_part_' + str(i + 1) + '.pkl'))

        return


class LDAPreparation(object):
    # self.data is the previous 'lda_df_full'!
    def __init__(self, load_path=None, raw_data=None, n_jobs=2, cs_threshold=0.5, output_type='All', seed=1,
                 spacy_model='en_core_web_sm', ngram_min_count=10, ngram_threshold=300):
        self.load_path = load_path
        self.raw_data = raw_data
        if self.load_path is not None and self.raw_data is not None:
            raise ValueError("Please give a value for 'load_path' or 'raw_data' (but not both)")
        if self.load_path is None and self.raw_data is None:
            raise ValueError("Please specify a value for 'load_path' or 'raw_data' (but not both)")
        if self.raw_data is not None:
            self.data = raw_data.reset_index()  # reset the index column from previous step.
        else:
            self.data = self.loading()
        self.cs_threshold = cs_threshold
        self.n_jobs = n_jobs
        self.ngram_min_count = ngram_min_count
        self.ngram_threshold = ngram_threshold
        self.output_type = output_type
        self.seed = seed
        self.spacy_model = spacy_model

    def loading(self):
        pickles = []
        for filename in glob.glob(os.path.join(self.load_path, '*.pkl')):  # check the data in folder
            pickles.append(pd.read_pickle(filename))

        lda_df_full = pd.concat(pickles, sort=False, ignore_index=True)  # concat all pkls
        return lda_df_full

    def hashtag_pooling(self):
        self.data = self.data.loc[:,
                    ['created_at', 'text', 'text_tokens', 'hashtags', 'center_coord_X', 'center_coord_Y']]
        print('Length of input data: ' + str(len(self.data)))

        rm.seed(self.seed)  # set seed

        # set from which the LDA is going to be trained
        print('Length of set: ' + str(len(self.data)))
        lda_df_ht = self.data[self.data.loc[:, 'hashtags'].str.len() > 0]  # get all entries with hashtag(s)
        lda_df_wht = self.data[self.data.loc[:, 'hashtags'].str.len() == 0]  # get all entries WITHOUT hashtag(s)

        # lowercase the hashtags:
        with warnings.catch_warnings():  # suppress warnings.
            warnings.simplefilter("ignore")
            lda_df_ht.loc[:, 'hashtags'] = lda_df_ht['hashtags'].apply(lambda x: str(x).lower())

        # Goal: pool hashtags for all already existing hashtags:

        # __Details about the following block:__
        # The next steps, start with loading the spacy model. Afterwards, the self-defined function tokenize_hashtags is placed. It basically works as the tokenizer functions described before in Appendix B but filters the tokens for punctuation, in this case the “#”-symbol and “\”-symbol, and removes symbols and spaces. The results are hashtags only consisting of alphanumeric characters. Then, the self-defined function unique_words is provided. This function, fed a nested list of strings, is returning a list of all the unique strings in the passed nested list. This is achieved by first looping over all nested lists and extending them into an empty list, called unique_words_list. unique_words_list = list(set(unique_words_list))) contains all unique strings and saves them as a single list, being the return value of the function.
        # Tokenize all hashtags in the 'lda_df_ht'-column, remove '#', spaces and punctiation. Additionally, get a list of all in the df exisisting hashtags, exactly once:
        # tokenize the hashtags and put them uniquely in a list:

        # loading the statistical spacy-model:
        nlp = spacy.load(self.spacy_model)

        # Predicts part-of-speech tags, dependency labels, named entities and more.
        def tokenize_hashtags(lis):
            doc = nlp(lis)  # list of hashtags
            test_list = []
            for token in doc:
                if (token.pos_ != 'PUNCT') & (token.text != '#') & (
                        token.text != '\''):  # remove '#', spaces and punctuation
                    # print(token.text)
                    test_list.append(token.text)

            return str(test_list)

        # get all the available hashtags once:
        def unique_words(lis):
            unique_words_list = []
            for i in lis:
                unique_words_list.extend(i)  # concat all lists
            unique_words_list = list(set(unique_words_list))  # remove duplicates and return every hashtag once
            return unique_words_list

            # __Details about the following blocks:__
            # Hereupon, prep, a nested list containing the hashtags for every tweet, is created by applying tokenize_hashtag on lda_df_ht[‘hashtags’] plus some further formatting. It will become a helper list to build the other two following lists. Consequently, prep_series is created, a pandas Series that contains the content of prep but additionally includes the corresponding indices from lda_df_ht. Finally, prep is passed to the unique_words function, the result is named prep_unique_hashtags.
            #
            # The following step is pooling all the tweets indices, where the tweet contains one of the unique hashtags. The self-defined function pooling_indices_to_hashtag takes two arguments: The first argument, unique_hashtags, is the list of unique words (here, applied on prep_unique_hashtags) and the second argument, index_hashtag_series, a Series (here, applied on prep_series). A loop runs over unique_hashtags. For every unique hashtag, it is now checked in which row it appears in index_hashtag_series. These results, a boolean list of the length of index_hashtag_series, are saved in hashtag_check. In res, the current hashtag is appended, as well as the index from index_hashtag_series, where hashtag_check equals True. After looping over the whole list of unique_hashtags, res is returned, containing nested lists with a unique hashtag, followed by a nested list of the corresponding indices of the tweets, which contains this hashtag. This is continued over all tweets to get all unique hashtags. The function is initialized with prep_unique_hashtags and prep_series and the result does overwrite prep. It is consequently transformed into a DataFrame named pooled, which now contains the hashtags and indices as columns.

            # **prep: nested:** list including the hashtags for every tweet (helper list to build the other two following).
            # <br>
            # **prep_series:** pd.Series including the index of the tweet in 'lda_df_ht' original df and the hashtags for every tweet.
            # <br>
            # **prep_unique_hashtags:** list of all in 'lda_df_ht' existing hashtags, exactly once.

        prep = lda_df_ht['hashtags'].apply(
            lambda x: tokenize_hashtags(x))  # apply fct on all rows of 'hashtags'-column
        prep_index_list = prep.index.values.tolist()
        prep = list(prep.apply(
            lambda x: ast.literal_eval(x)))  # convert series-list-LIKE-elements to list instead of string
        # as if using series.tolist() or list(series)
        prep_series = pd.Series(prep)  # Serialize tokenized hashtags
        prep_series.index = prep_index_list  # append the corresponding indices from 'lda_df_ht' to the Series of tokens.

        prep_unique_hashtags = unique_words(prep)  # get all the available hashtags once.

        # combine unique hashtags with all corresponding indices
        def pooling_indices_to_hashtag(unique_hashtags, index_hashtag_series):
            res = []
            for i in unique_hashtags:
                hashtag_check = index_hashtag_series.apply(
                    lambda x: i in x)  # check in which columns the hashtag appears
                res.append(i)  # append the current hashtag
                res.append(
                    list(index_hashtag_series[hashtag_check == True].index))  # append it to the corresponding indices
            return res

        prep = pooling_indices_to_hashtag(prep_unique_hashtags, prep_series)

        # turn the result into an dataframe
        pooled = pd.DataFrame({'hashtag': prep[::2], 'index': prep[1::2]})

        # __Details about the following blocks:__ 
        # Next, the self-defined function pool_tweets is taking care of pooling the text tokens using the indices in prep. To achieve this, a nested for-loop is used to first go over the main list and then through the sub-lists, containing the desired indices. J_token stores the corresponding tokens by accessing them from lda_df_ht by going through every element of the nested indices lists. The tokens of every index saved in j_token are appended to group_of_tokens, an empty list created in every round of the outer loop. This list itself is appended to all_tokens, an empty list created outside of the loops. All_tokens is returned by the function. Thereby it is achieved that all_tokens becomes a nested list containing the tokens of all pooled tweets for every unique hashtag. The function is applied on every second row of prep, containing the lists of indices and is saved as pooled[‘pooled_tweets_token’], which contains all the necessary information, now. 

        # get the tokens from the 'lda_df_ht'-indices
        def pool_tweets(indices):
            all_tokens = []
            row = 0
            for i in indices:
                group_of_tokens = []
                for j in range(len(i)):
                    j_token = lda_df_ht.loc[indices[row][j], 'text_tokens']
                    group_of_tokens.extend(j_token)
                row = row + 1
                all_tokens.append(group_of_tokens)
            return all_tokens

        # append the tokens to its corresponding hashtags and indices
        pooled['pooled_tweets_token'] = pool_tweets(prep[1::2])

        # # Calculate Cosine Similarity to append unlabeled Tweets to our Hashtags:

        # After the Tweets were pooled via unique hashtag (tweets with more than one hashtag were appended to all of the “hashtag-groups”), the cosine-similarity (measure for the similarity of two vectors) between an unlabeled tweet and all the hashtag-pools is now going to be calculated, based on TF-IDF. Unlabeled tweets that passed a certain threshold are appended to the hashtag-pool, which whom they have the highest score.

        # The following cells are embedded into try-except blocks for the case that there were no tweets without hashtag passed!

        # __Details about the following block:__ 
        # The script continues by defining all_tweets_pool, which concatenates pooled with the corresponding information from all the tweets without a hashtag. Then, all tweets without a hashtag are concatenated, leading to a DataFrame that looks like pooled on the upper entries but ends with the entries from all tweets without a hashtag (single tweets). Obviously, the all_tweets_pool[‘hashtag ‘] column only consists of an empty list if an entry is a single tweet. The same applies to all_tweets_pool[‘index‘], which has only one value on each row, logically.

        # append all unlabeled tweets 'lda_df_wht' indices and tokens to the 'pooled' df:
        # try:
        # append all unlabeled tweets 'lda_df_wht' indices and tokens to the 'pooled' df:
        tweets_without = lda_df_wht.loc[:, ['hashtags', 'text_tokens']]
        tweets_without['index'] = lda_df_wht.index
        tweets_without = tweets_without[['hashtags', 'index', 'text_tokens']]
        tweets_without.columns = ['hashtag', 'index', 'pooled_tweets_token']
        tweets_without['index'] = tweets_without['index'].apply(lambda x: [x])

        all_tweets_pool = pd.concat([pooled, tweets_without], ignore_index=True)

        # except:
        #    pass

        # __Details about the following block:__
        # The goal is now to calculate the cosine similarity between every single tweet and the tweet pools. Using larger data samples, the processing load becomes extremely high, since every single tweet needs to be checked against every hashtag pool. The number of computations is hereby equal to the number of hashtag pools times the number of single hashtags. One can easily see that this causes a problem with larger samples.
        # To tackle this issue, the calculation of cosine similarity between the single tweets and tweet pools is parallelized. The single tweets are divided into sublists, the number is determined by the amount of worker processes that have been assigned.
        # 
        # First, the number of worker processes is to be defined by self.n_jobs while self.cs_threshold is defining the confidence threshold C. A tweets cosine similarity that is exceeding this threshold is attached to the hashtag pool. Second, pooled_to_vectorize is created, a list containing the “raw” tokens, only separated by whitespace instead of comma. Third, vecorizer_fit is constructed, a variable containing the TF-IDF values of all tweets, fitted by the TfidfVectorizer model from the scikit (sklearn) package. Followed by that, a self-written py-file is imported, called parallel_cs_2, containing a self-defined function called parallel. This function is later called by the worker processes. Next, the length of pooled is saved in len_pooled. Vecorizer_fit_unpooled stores all the TF-IDF values of the hashtag pools. no_of_packages_to_pass contains the number “packages” of  TF-IDF values of single tweets to be passed to the parallel function by one iteration. This number is determined by self.n_jobs.
        # 
        # The following initializes the parallelization: Pool is allocated to n parallel processes. Then, over the range of the length of no_of_packages_to_pass, an upper_bound and lower_bound are created. These values are determined by self.n_jobs and picking the current single tweets by index-slicing, which are transferred to parallel in one iteration of the outer loop. For example, if self.n_jobs = 20, then 20 TF-IDF values of single tweets from vecorizer_fit_unpooled, each as a sublist, are passed to the single_vectorizer_fit_unpooled_values list. This passing of the single tweets into single_vectorizer_fit_unpooled_values, together with a counter, counting each single tweets position from zero, is done by the inner for-loop. The parallel processes are working on one of these sub-lists, each.     
        # 
        # The function pool.map takes a function and an iterator. The function is passed via partial, which appends an argument to a given function. Here, the function parallel is passed with all its arguments, except one. The last argument, single_vectorizer_fit_unpooled_values, serves as the iterator for pool.map. Pool.map is simply applying parallel n-times by iterating the processes of pool over single_vectorizer_fit_unpooled_values. Since partial is fixing all the other arguments of parallel, the function is called self.n_jobs-times in parallel with the same values for all arguments, except single_vectorizer_fit_unpooled_values, which always passes a different sublist of itself. The return value for every iteration is saved in res, which is extended into the empty result list final_res on every iteration. pool.close() closes the worker processes after all packages are passed and all results are returned. 

        # __Additional information about *parallel_cs_2.py*:__
        # Let’s now have a look at parallel, the function that is called n-times in parallel. It is saved in the py-file parallel_cs_2. The reason for this is that jupyter as an interactive interpreter cannot call a function in a current session parallel because of the global interpreter lock. Therefore, the function is saved outside of the current session and is then reloaded.
        # Parallel takes the following, above described, arguments: pooled_to_vectorize, self.cs_threshold, len_pooled, vectorizer_fit and single_vectorizer_fit_unpooled_values. After that, every worker thread takes one TF-IDF value from a single unlabeled tweet in form of a sublist from single_vectorizer_fit_unpooled_values and calculates the cosine similarity of all TF-IDF values of the hashtag pools and its passed TF-IDF value. The results are sorted by size and the largest one is saved in most_similar_tweets_index. Next, it is checked if this value can cross the self.cs_threshold. If True, the counter value of the single tweet is appended to indices_unlabeled, the corresponding cosine similarity score is appended to value_of_cs and the index number of the hashtag pool from pooled_to_vectorized (where the single tweet shall be appended later) is appended to indices_to_append. This process is repeated over the range of the length of no_of_packages_to_pass. The function finally returns indices_unlabeled, value_of_cs, indices_to_append_atp, if the single tweets TF-IDF is greater than the self.cs_threshold, else it returns None. The returned values are stored in res.    
        # 

        # prepare the token-sets with hashtags:
        pooled_to_vectorize = all_tweets_pool['pooled_tweets_token'].apply(
            lambda x: ' '.join(x))  # nested list of all tokens, 
        # not separated by comma anymore but only by whitespace.
        vectorizer = TfidfVectorizer()  # computes the word counts, IDF values, and Tf-idf scores all using the same dataset.
        vectorizer_fit = vectorizer.fit_transform(pooled_to_vectorize)  # fit the "TfidfVectorizer()"-model

        len_pooled = len(pooled)  # len of 'pooled'
        vectorizer_fit_unpooled = vectorizer_fit[len(pooled):]  # all fitted TF-IDF values for the tweet pools

        no_of_packages_to_pass = math.ceil(
            len(pooled_to_vectorize[len(pooled):]) / self.n_jobs)  # calculate the amount of packages of 
        # TF-IDF values to be passed.

        final_res = []
        upper_bound = self.n_jobs
        counter = 0
        # parallelization:
        pool = Pool(processes=self.n_jobs)  # initialize parallelization

        for i in range(0, no_of_packages_to_pass):
            lower_bound = upper_bound - self.n_jobs

            if i % 100 == 0:
                print("Total number of loop iterations for hashtag-pooling: " + str(no_of_packages_to_pass))
            single_vectorizer_fit_unpooled_values = []  # put values of sparse matrix for every single tweet into a nested list, with number of nested lists == self.n_jobs
            for j in range(lower_bound, upper_bound):
                try:
                    single_vectorizer_fit_unpooled_values.append([vectorizer_fit_unpooled[j], counter])
                except:
                    break
                counter = counter + 1  # count index of "tweets_without", where the tested single tweet is stored

            try:
                # print("cpu count: " + str(cpu_count()))
                # print('pool count: ' + str(pool._processes))
                res = pool.map(partial(parallel, pooled_to_vectorize, self.cs_threshold, len_pooled,
                                       vectorizer_fit),
                               single_vectorizer_fit_unpooled_values)

                # pool.map: take a function and an iterator to parallelize over. function arguments passed via 'partial' stay
                # fixed via parallelization. 'nested_list_of_vectorizer_fit_single_tweets' is iterated over and worked on parallel.
                # parallel_cs.parallel: function to be parallelized over. It is stored in a separate py-file, parallel_cs.py,            #because jupyter is an interctive interpreter and can't call a function out of the same session, if parallelization is applied.
                # for more information see: https://stackoverflow.com/questions/20222534/python-multiprocessing-on-windows-if-name-main

            except KeyboardInterrupt:
                pool.terminate()  # terminate worker processes in case of keyboard interrupt
                break

            # except Exception as e:
            #   pool.terminate()
            #   if hasattr(e, 'message'):
            #       print(e.message)
            #   else:
            #       print(e)

            upper_bound = upper_bound + self.n_jobs
            final_res.extend(res)  # save the results
            print('current loop:', i)  # print current loop iteration (progress)

        pool.close()  # close worker processes
        final_res = list(filter(None, final_res))

        # __Details about the following blocks:__
        # The three returned lists for every fitting single tweet are stored as nested lists in final_res.
        # For that reason, the values of final_res are transformed using indexing to get them into the shape of three
        # simple lists, named the same as the return values of parallel.
        # Finally, these values are used to get the index of every single tweet and extend it to the indices of the
        # hashtag pool. The same is done with the single tweets tokens, which are extended into the hashtag pools tokens
        # as well. A new DataFrame lda_all_tweets_pooled is created, containing the updated hashtag pools, which are
        # now the final pseudo-documents, including all single tweets with one hashtag. The remaining non-pooled,
        # single tweets without a hashtag are discarded. This data is now ready to train the LDA.

        # get 'final_res' into the right form:
        indices_unlabeled = []
        value_of_cs = []
        indices_to_append_atp = []

        for i in range(0, len(final_res)):
            indices_unlabeled.append(final_res[i][0])
            value_of_cs.append(final_res[i][1])
            indices_to_append_atp.append(final_res[i][2])

        indices_unlabeled = list(it.chain(*indices_unlabeled))
        value_of_cs = list(it.chain(*value_of_cs))
        indices_to_append_atp = list(it.chain(*indices_to_append_atp))

        tweets_without = tweets_without.reset_index(
            drop=True)  # reset the index of 'tweets_without' to adapt it to the indexing scheme of 'indices_unlabeled'

        all_appended_single_tweets = []
        for i in range(0, len(indices_unlabeled)):
            new_index = pooled.iloc[indices_to_append_atp[i]][1]  # get the indices of the matching pool
            new_index.append(
                tweets_without.iloc[indices_unlabeled[i]][1])  # append the indices by the one to be appended
            pooled.iloc[indices_to_append_atp[i]][1] = new_index  # overwrite the old indices

            all_appended_single_tweets.append(
                tweets_without.iloc[indices_unlabeled[i]][1])  # get indices of all appended
            # single tweets

            new_token = pooled.iloc[indices_to_append_atp[i]][2]  # get the old token-list of the hashtag
            new_token.extend(tweets_without.iloc[indices_unlabeled[i]][2])  # extend the cs-unlabeled matches' tokens
            pooled.iloc[indices_to_append_atp[i]][2] = new_token

        if self.output_type == 'pool_hashtags':  # return pools plus hashtagged tweets
            lda_all_tweets_pooled = pooled
        elif self.output_type == 'pool':  # return only pooled tweets
            lda_all_tweets_pooled = pooled[pooled['index'].str.len() > 1]
        else:  # return all tweets (remove the appended single tweets beforehand)
            final_single_tweets = tweets_without[~pd.Series(list(it.chain(*tweets_without['index'])))
                .isin(list(it.chain(*all_appended_single_tweets)))]
            frames_to_concat = [pooled, final_single_tweets]
            lda_all_tweets_pooled = pd.concat(frames_to_concat)

        # __Details about the following blocks:__
        # Using the Phrases and Phraser functions of the gensim package,
        # the tokens of lda_all_tweets_pooled['pooled_tweets_token'] are transformed to bigram tokens.
        # The Phrases function lets one specify a scoring threshold for forming the bigram phrases.
        # The actual creation of the bigrams is implemented by the self-defined function make_bigrams.
        # The result is stored as a new column: lda_all_tweets_pooled[‘bi_grams’]. Next, a gensim dictionary
        # is created to store the vocabulary of the corpus, named dic_id2word_bi. It is afterwards filtered
        # to get the DTM of the LDA models in the desired shape. Only the 50.000 (keep_n=50000)
        # most occuring words and only words that occur at least 10-times (no_below=10) are saved.
        # Additionally, no words that occur in more than 85% (no_above=0.85) of the documents are saved.
        # This filtering is necessary to improve the quality and training performance of the LDAs.
        # Finally, the corpus for the LDA is created using lda_all_tweets_pooled[‘bi_grams’], the bigrams of the pooled
        # tweets. Corpus_bi is containing all documents created by the bigrams which themselves are created from the pooled tweets.
        # Now, everything is finally ready for LDA training.
        #


        # example:
        # print(bigram_mod[lda_all_tweets_pooled['pooled_tweets_token'].iloc[3]])

        # make_(bi-)trigrams functions are provided as static methods!
        lda_all_tweets_pooled['bi_grams'] = LDAPreparation.make_ngrams(lda_all_tweets_pooled['pooled_tweets_token'],
                                                                        self.ngram_min_count, self.ngram_threshold)
        lda_all_tweets_pooled['tri_grams'] = LDAPreparation.make_ngrams(lda_all_tweets_pooled['pooled_tweets_token'],
                                                                          self.ngram_min_count, self.ngram_threshold,
                                                                          ngram_type='tri')

        def trylambda(value): #flatten single tweets indices
            try:
                return list(it.chain(*value))
            except (ValueError, TypeError):
                return value

        lda_all_tweets_pooled['index'] = lda_all_tweets_pooled['index'].apply(lambda x: trylambda(x))
        pd.set_option('display.max_rows', 5000)
        print(lda_all_tweets_pooled['index'])
        self.lda_all_tweets_pooled = lda_all_tweets_pooled #all tweets which are used for the next steps

        return

    def lda_training(self, data_save_path, models_save_path, data_save_type='pkl', ngram_style='unigrams',
                     filter_keep_n=15000, filter_no_below=10,
                     filter_no_above=0.85, topic_numbers_to_fit=[10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300],
                     n_saved_top_models=3):

        if ngram_style == 'unigrams':
            ngram_type = 'pooled_tweets_token'
        elif ngram_style == 'bigrams':
            ngram_type = 'bi_grams'
        elif ngram_style == 'trigrams':
            ngram_type = 'tri_grams'
        else:
            return print('This is not a valid choice for "ngram_style"! Choose between "unigrams" (default), "bigrams"'
                         'and "trigrams"!')

        def dic_corpus_creation(ngram_type):
            # source: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/7
            # Create Dictionaries:
            dic_id2word = corpora.Dictionary(self.lda_all_tweets_pooled[ngram_type])
            # filter the dictionary, see:
            # https://radimrehurek.com/gensim/corpora/dictionary.html#gensim.corpora.dictionary.Dictionary.filter_extremes
            # no_below (int, optional) – Keep tokens which are contained in at least no_below documents.
            # no_above (float, optional) – Keep tokens which are contained in no more than no_above documents (fraction of total corpus size, not an absolute number).
            # keep_n (int, optional) – Keep only the first keep_n most frequent tokens.
            # keep_tokens (iterable of str) – Iterable of tokens that must stay in dictionary after filtering.
            dic_id2word.filter_extremes(keep_n=filter_keep_n, no_below=filter_no_below, no_above=filter_no_above)
            print('Raw vocabulary size before filtering extreme tokens:' + str(len(dic_id2word)))
            # Create Corpus: Term-Document Frequency for every tweet-pool
            corpus = [dic_id2word.doc2bow(tweets_pooled) for tweets_pooled in self.lda_all_tweets_pooled[ngram_type]]
            return dic_id2word, corpus

        dic_id2word, corpus = dic_corpus_creation(ngram_type)

        # # Train LDAs and get the optimal number of Topics by topic coherence analysis:

        # __Details about the following blocks:__
        # This segment executes the training of LDAs with different topic numbers and chooses the best three models for further usage by coherence value comparison.
        #
        # First, as always, n_jobs defines the number of CPU-cores to use. Train_a_lda_and_compute_coherence_values takes four arguments starting with corpus, which is always the variable corpus_bi, and id2word, which is the above defined gensim dictionary dic_id2word_bi. Num_topics gets a list called topic_numbers_to_fit assigned, which contains integers stating the topic number for every LDA, and is later iterated over to try out different numbers of topics. The last argument is n_jobs.
        # The function trains an LDA and calculates its coherence, returning the coherence value, the number of topics of the model and the fitted LDA model. The LDA model is fitted by lda = gensim.models.ldamulticore.LdaMulticore(...). The function fits the LDA using a Variational Bayes algorithm for approximation, based on Hoffman et al. (2010). This algorithm can optimize posterior approximation time, leading to reduced computational time in comparison to traditional batch algorithms, when it comes to large datasets. The algorithm is used in models.ldamulticore as well as in models.ldamodel (Gensim 2020b, 2020c).
        # The used configuration of the LdaMulticore method stays almost default, since we want to keep things as general as possible and do not want to include prior information regarding topic and / or word probability. The asymmetric prior of topic and word distribution are learned directly from the data (Gensim 2020b, 2020c). The corpus is iterated over 30 times during training, which is set relatively high to ensure document convergence.
        #
        # After fitting the a LDA model, it is transferred to coherence_model_lda = CoherenceModel(model=lda, texts=lda_all_tweets_pooled['bi_grams'],  dictionary=id2word, coherence='c_v'), where the lda models topic coherence is calculated. As already stated above, Train_a_lda_and_compute_coherence_values returns the coherence value of the model, the number of topics of the model and the fitted LDA model itself. It is applied on a large variety of different topic numbers, passed to it by topic_numbers_to_fit. The models with the three highest c_v scores are then selected and used for the topic distribution calculation. They are subsequently saved.
        #
        #

        # Train LDA Models:
        # sources: https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
        # https://medium.com/datadriveninvestor/nlp-with-lda-analyzing-topics-in-the-enron-email-dataset-20326b7ae36f
        # https://github.com/xillig/nlp_yelp_review_unsupervised/blob/master/notebooks/2-train_corpus_prep_and_LDA_train.ipynb

        # see: https://radimrehurek.com/gensim/models/ldamulticore.html
        def train_a_lda_and_compute_coherence_values(corpus, id2word, num_topics, n_jobs):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lda = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                                              # corpus — Stream of document vectors or sparse matrix of shape (num_terms, num_documents)
                                                              id2word=id2word,
                                                              # id2word – Mapping from word IDs to words.
                                                              # It is used to determine the vocabulary size, as well as for debugging and topic printing.
                                                              num_topics=num_topics,
                                                              # num_topics — The number of requested latent topics to be extracted from the training corpus.
                                                              random_state=100,
                                                              # random_state — Either a randomState object or a seed to generate one. Useful for reproducibility.
                                                              # update_every=1,
                                                              # update_every — Number of documents to be iterated through for each update.
                                                              # Set to 0 for batch learning, > 1 for online iterative learning.
                                                              # NOT IN Lda.Multicore!!!
                                                              chunksize=60000,
                                                              # chunksize — Number of documents to be used in each training chunk.
                                                              workers=n_jobs,
                                                              # workers: number of physical cpu-cores. use core-number - 1
                                                              passes=30,
                                                              # passes — Number of passes through the corpus during training.
                                                              # alpha='auto',
                                                              # alpha — auto: Learns an asymmetric prior from the corpus
                                                              # NOT IN Lda.Multicore!!!
                                                              per_word_topics=True)
                # per_word_topics — If True, the model also computes a list of topics, sorted in descending order of most
                # likely topics for each word, along with their phi values multiplied by the feature-length (i.e. word count)

            # calculate c_v coherence. see: https://radimrehurek.com/gensim/models/coherencemodel.html and http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf
            coherence_model_lda = CoherenceModel(model=lda, texts=self.lda_all_tweets_pooled[ngram_type],
                                                 dictionary=id2word, coherence='c_v')

            return [coherence_model_lda.get_coherence(), num_topics,
                    lda]  # return the coherence value, number of topics, and the fitted lda model

        res = []
        print('Start LDA training!')
        print(datetime.now())
        for i in topic_numbers_to_fit:
            res.append(train_a_lda_and_compute_coherence_values(corpus, id2word=dic_id2word, num_topics=i,
                                                                n_jobs=self.n_jobs))
            print(i)
            print(datetime.now())

        # sort results by best coherence score:
        coherence_values = []
        index_of_best = []
        best_models = []
        for i in range(len(res)):  # get the coherence values of all models
            coherence_values.append(res[i][0])
        for i in nlargest(n_saved_top_models, coherence_values):  # choose the n-best models
            index_of_best.append(coherence_values.index(i))
        for i in index_of_best:  # get all information of the best n models
            best_models.append(res[i])

        models = {}  # save the models for usage in a dictionary
        for i in best_models:
            models['lda_' + str(i[1]) + '_topics_' + ngram_style] = i[2]

        # Define function to save the n models, vocabulary and corpus:
        def save_lda_models(model, save_name, path):
            p = (path, save_name)
            d_path = os.path.join(*p)  # join path and name, "datapath()" is unabele to join them
            d_path = datapath(d_path)
            model.save(d_path)  # save the model to specified path
            return

        # save models:
        for key, value in models.items():  # save the models by using theier dictionary named(models.keys)
            save_lda_models(value, key, models_save_path)

        # save vocabulary:
        with open(os.path.join(models_save_path, 'dic_id2word_' + ngram_style + '.pkl'), 'wb') as f:
            pickle.dump(dic_id2word, f)

        # save corpus:
        with open(os.path.join(models_save_path, 'corpus_' + ngram_style + '.pkl'), 'wb') as f:
            pickle.dump(corpus, f)

        # __Details about the following blocks:__
        # The next part describes the preparation of the single tweets for the calculation of the topic distributions
        # by the trained LDA models. Indices_of_pooled_unique_tweets is a list, where all indices of all tweets are
        # saved that were used during the pooling process. Making use of this list, a new DataFrame is built from
        # lda_df_full called lda_df_trained_tweets, containing all information from the pooled tweets. Next, bigrams
        # are created in the same way as above for this DataFrame. Hereby, it is important to use the pretrained
        # dic_id2word_bi gensim dictionary since the resulting corpus ut_corpus_bi is later passed to the pre-trained
        # LDAs for topic distribution calculation. Using this gensim dictionary, it is ensured that the corpus is
        # formed only by words the LDA was trained on. Doc2bow is a utility function used for a sparser vector
        # representation of the corpus. The same procedure is done for the test set lda_df_test_set, resulting in the
        # corpus test_set_corpus_bi.

        # get a set of all, for-pooling-used tweets:
        indices_of_pooled_unique_tweets = []
        print(self.lda_all_tweets_pooled)
        print(len(self.lda_all_tweets_pooled))

        for i in range(len(self.lda_all_tweets_pooled)):
            if type(self.lda_all_tweets_pooled['index'].iloc[i]) is list: #in case index is from as single appended tweet
                print(self.lda_all_tweets_pooled['index'].iloc[i])
                indices_of_pooled_unique_tweets.extend(self.lda_all_tweets_pooled['index'].iloc[i])
            else:
                print(self.lda_all_tweets_pooled['index'].iloc[i])
                indices_of_pooled_unique_tweets.extend(self.lda_all_tweets_pooled['index'].iloc[i])
        indices_of_pooled_unique_tweets = list(set(indices_of_pooled_unique_tweets))  # get only the unique tweets

        # make a COPY of df of all for lda-training used tweets
        lda_df_trained_tweets = self.data.loc[indices_of_pooled_unique_tweets, :].copy()

        # As for the pooled tweets, get also bigrams for the the tweets that produced the pooled tweets.

        # create corpus for the training set:
        if ngram_type == 'bigrams':
            lda_df_trained_tweets['bi_grams'] = LDAPreparation.make_ngrams(lda_df_trained_tweets['text_tokens'],
                                                                           self.ngram_min_count, self.ngram_threshold)
        if ngram_type == 'trigrams':
            lda_df_trained_tweets['tri_grams'] = LDAPreparation.make_ngrams(lda_df_trained_tweets['text_tokens'],
                                                                            self.ngram_min_count, self.ngram_threshold,
                                                                            ngram_type='tri')

        # IMPORTANT! use the pre-trained "dic_id2word*"-objects!
        if ngram_type == 'pooled_tweets_token':
            ut_corpus = [dic_id2word.doc2bow(tweets_unique) for tweets_unique in lda_df_trained_tweets['text_tokens']]

        if ngram_type == 'bigrams':
            ut_corpus = [dic_id2word.doc2bow(tweets_unique) for tweets_unique in lda_df_trained_tweets['bi_grams']]

        if ngram_type == 'trigrams':
            ut_corpus = [dic_id2word.doc2bow(tweets_unique) for tweets_unique in lda_df_trained_tweets['tri_grams']]

        # __Details about the following blocks:__
        # The next part finally outlines the single tweets topic distribution calculation for the training and test set, lda_df_trained_tweets and lda_df_test_set. The self-defined function get_topics_for_tweets is able to calculate the topic distribution for every tweet in a passed set by using a passed LDA model. In this case, only bigrams are used, but it is also possible to calculate the topic distributions for single tokens or trigrams, by setting gram_type = ‘token’ or gram_type = ‘tri’, respectively. A counter shows the calculation progress. Topic distribution calculation is done by the lda_model.get_document_topics function. Minimum_probability=0.0 assures that every topic probability is returned, regardless of how low it might be. Respectively, calculation is done by using the corpi ut_corpus_bi or test_set_corpus_bi. The top_topics are then saved in topic_vec and afterwards supplemented by the number of tokens (including bigram tokens) of the respective tweet (word count) by using topic_vec.extend([len(lda_df_trained_tweets['token_tm'].iloc[i])]). Iterating over all tweets, every tweets topic_vec is appended to topic_vecs, a nested list containing the topic distributions of all tweets of the dataset. This list is then returned by the function.
        #
        # Get_topics_for_tweets is applied on each of the three best performing models by iterating over models, meaning both the training as well as the test set are getting three computed topic distributions with different topic numbers, each. The results are automatically appended to the lda_df_trained_tweets and lda_df_test_set DataFrames as new columns and are named accordingly. Once the data is appended, the two Dataframes are saved.

        # Finally, compute the topic distribution for every tweet from the by the trained LDA-model:

        # get the topic distribution for all the tweets in the training and test set.
        # source: https://github.com/xillig/nlp_yelp_review_unsupervised/blob/master/notebooks/2-train_corpus_prep_and_LDA_train.ipynb

        def get_topics_for_tweets(lda_model, number_of_topics):
            train_vecs = []
            counter = 0
            for i in range(len(lda_df_trained_tweets['text_tokens'])):
                if counter % 500 == 0:
                    print(counter)
                top_topics = lda_model.get_document_topics(ut_corpus[i],
                                                            minimum_probability=0.0)  # calculate the topic distribution for every tweet in test set
                topic_vec = [top_topics[i][1] for i in
                             range(number_of_topics)]  # get the distribution values for all topics
                topic_vec.extend(
                    [len(lda_df_trained_tweets['text_tokens'].iloc[i])])  # include length of tweet as covariate, too
                train_vecs.append(topic_vec)
                counter = counter + 1

            return train_vecs

        list_of_topic_distr_trained_tweets_lda = []
        counter = 0
        for lda_model in models.values():
            list_of_topic_distr_trained_tweets_lda.append(
                get_topics_for_tweets(lda_model=lda_model, number_of_topics=best_models[counter][1]))
            counter = counter + 1

        # save the data:
        for i, j in zip(range(len(list_of_topic_distr_trained_tweets_lda)),
                        models.keys()):  # iterate over all models topic distribution set and get the name
            # put the values of the topic distributions of the trained tweets to its df
            lda_df_trained_tweets[str(j)] = pd.Series(list_of_topic_distr_trained_tweets_lda[i]).values

        # save training set topic distribution
        _pack_size = 250000  # no of tweets saved in one go

        parts_to_save = math.ceil(len(lda_df_trained_tweets) / _pack_size)  # calculate how many parts to save

        upper_bound = _pack_size
        for i in range(0, parts_to_save):
            lower_bound = upper_bound - _pack_size
            file_to_save = lda_df_trained_tweets.iloc[lower_bound:upper_bound, :]
            upper_bound = upper_bound + _pack_size

            if data_save_type == 'csv':
                file_name = 'my_lda_df_trained_tweets_save_part_' + str(i + 1) + '.csv'
                file_to_save.to_pickle(os.path.join(data_save_path, file_name))
            else:
                file_name = 'my_lda_df_trained_tweets_save_part_' + str(i + 1) + '.pkl'
                file_to_save.to_csv(os.path.join(data_save_path, file_name))

        return




    # Building ngrams:
    # source: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

    # functionalize ngramization:
    @staticmethod
    def make_ngrams(corpus, ngram_min_count, ngram_threshold, ngram_type=None):
        # Build the bigram model:
        # source: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
        bigram = Phrases(corpus, min_count=ngram_min_count,
                         threshold=ngram_threshold)  # higher threshold -> fewer phrases.
        #trigram-model:
        if ngram_type == 'tri':
            trigram = Phrases(bigram[corpus], min_count=ngram_min_count,
                            threshold=ngram_threshold)

        # Faster way to get a sentence clubbed as a bigram
        bigram_mod = Phraser(bigram)
        if ngram_type == 'tri':
            trigram_mod = Phraser(trigram)
            return [trigram_mod[bigram_mod[hashtag_pool]] for hashtag_pool in corpus]
        return [bigram_mod[hashtag_pool] for hashtag_pool in corpus]



# parallel called function for cosine similarity calculation:
def parallel(pooled_to_vectorize, cs_threshold, len_pooled, vectorizer_fit, single_tweet_vectorizer_fit_unpooled):
    # pooled_to_vectorize: nested list of all tokens, not separated by comma anymore but only by whitespace.
    # cs_threshold: cosine similarity threshold. see markdown explanation about "Hashtag Labeling Algorithm" in script "LDA Preparation".
    # len_pooled: length of all hashtag-pooled tweets.
    # vectorizer_fit: fitted TF-IDF values for all tweets
    # single_tweet_vectorizer_fit_unpooled: list containing fitted TF-IDF values of the single (unpooled) tweets. no. of entries is defined by no. of workers
    indices_unlabeled = []
    value_of_cs = []
    indices_to_append_atp = []

    # check all hashtag pool vs. one single tweet at a time at any worker.
    cs = cosine_similarity(vectorizer_fit[:len_pooled], single_tweet_vectorizer_fit_unpooled[0]).flatten()

    # gets the index with the highest cs score for a pooled hashtag.
    most_similar_tweets_index = cs.argsort()[:-2:-1]

    if cs[most_similar_tweets_index] > cs_threshold:  # cs min. default: 0.5
        indices_unlabeled.append(
            single_tweet_vectorizer_fit_unpooled[1])  # index number of unlabeled tweet in 'all_tweets_pool'
        value_of_cs.append(cs[most_similar_tweets_index][0])  # corresponding cs value
        indices_to_append_atp.append(pooled_to_vectorize.index[most_similar_tweets_index][
                                         0])  # index number of hashtag pool in 'all_tweets_pool', where the single tweet shall be appended.

        return indices_unlabeled, value_of_cs, indices_to_append_atp

    else:
        return  # return 'None' if cs of single tweet couldn't pass the threshold



##################################################################################################

#import sys
#import pandas as pd
#sys.path.append(r'C:\Users\gilli\PycharmProjects\TopViz\self-defined classes') #add path to be able to import produced classes.
#from data_scraping_classes import WriteFileListener
#from data_cleaning_location_classes import Cleaner
#from data_cleaning_location_classes import LDAPreparation

#pd.set_option('display.max_columns', None)  # show all columns

#wfl = WriteFileListener(r'C:\Users\gilli\OneDrive\Dokumente\Uni\Masterarbeit\Wichtige Informationen\Twitter API Access.txt',
#                            save_path=r'C:\Users\gilli\OneDrive\Desktop', languages=['en'],locations=[-125,25,-65,48],
#                        hashtag=False)

#c = Cleaner(load_path=r'C:\Users\gilli\OneDrive\Desktop')
#print(c.raw_data)
#c.saving(r'C:\Users\gilli\OneDrive\Desktop')

if __name__ == '__main__': #Mandatory for windows! see: https://stackoverflow.com/questions/58323993/passing-a-class-to-multiprocessing-pool-in-python-on-windows
    d = LDAPreparation(load_path=r'C:\Users\gilli\OneDrive\Desktop')
    #print(type(d.data))
    d.hashtag_pooling()
    d.lda_training(data_save_path=r'C:\Users\gilli\OneDrive\Desktop\test', models_save_path=r'C:\Users\gilli\OneDrive\Desktop\test',
                   ngram_style='bigrams', topic_numbers_to_fit=[10], n_saved_top_models=1)