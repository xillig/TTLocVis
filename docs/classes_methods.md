#Classes and Methods

This page provides an overview
of the classes and Methods provided by the TTLocVis
package.

##Class TwitterStreamer
The class *TwitterStreamer* provides the following functionalities:

- Streaming of tweets via the Twitter API.
- Saving collected tweets as JSON-file.

__Additional information:__
To be able to scrape Twitter data, one first needs to have access to it. Therefore, a Twitter Developer
Account is necessary to access the Twitter API. This account can be obtained through an application process
on the Twitter website, in which one has to enter his or her personal data and a description of the project
one plans to use the Twitter data with: https://developer.twitter.com/en <br>

```python
class TwitterStreamer(auth_path, languages, locations, save_path=os.getcwd(), extended=True,
                      hashtag=True)
```

__Parameters:__

- __auth_path (str):__ Path to the txt-file containing the users Twitter API credentials (see below).
- __languages (list of str):__ Language codes of desired language(s) of the streamed tweets content. See
 [Twitter language codes] for more info.
- __locations (list of float/int):__ Box-coordinates for streaming locations. Example: `[-125,25,-65,48]`
- __save_path (str):__ Path to where the json files shall be saved. Default is the working directory.
- __extended (bool):__ Decides if only *extended tweets* are collected. Default is `True`.
- __hashtag (bool):__ Decides if only tweets with a minimum of one hashtag are collected. Default is `True`.

[Twitter language codes]: https://developer.twitter.com/en/docs/twitter-for-websites/twitter-for-websites-supported-languages/overview

__Authentication procedure:__ After getting an Twitter developer account, one has to verify themselves
via the personal key, token and secrets. TwitterStreamer is using this information to access the Twitter API.
The Twitter API credentials shall be passed as a txt-file containing the necessary information
line by line in the following order:
*<br>consumer key<br> consumer secret<br> access token<br> access secret*<br>

__Note:__ The streaming process continues indefinitely until the user is shutting down the process by 
KeyboardInterrupt (Ctrl + C). Additionally, sometimes the direct filtering for *hashtag* and *extended tweet*
are permeable. Therefore, additional filtering is possible during the cleaning process described below.

## Class Cleaner
The class *Cleaner* provides the following functionalities:

- loading-in of Twitter-JSON-files
- Data cleaning: removal of duplicates, quoted tweets and retweets.
- Text cleaning: removal of Hyperlink-embeddings and mentions (usernames), identification and handling of hashtags and 
emojis.
- Handling location data: access bounding-box coordinates and calculate its center.
- Accessing user meta-data for every tweet.
- Removing unnecessary data. 
- Tokenization and lemmarization of the tweets text.
method __*saving*__:
- Saving the processed tweets as *pickle* or *csv* in batches of max. 300.000 tweets, each.

__Note:__ At the moment it is impossible to process data sets larger than 150.000 tweets in one go using this
 class due to performance and stability reasons!

```python
class Cleaner(load_path, data_save_name='my_cleaned_and_tokenized_data', languages=None,
              metadata=False, min_tweet_len=None, spacy_model='en_core_web_sm')
```
__Parameters:__

- __load_path (str):__ Path containing raw Twitter-JSON-files.
- __data_save_name (str):__ Name of the data saved to drive after processing (without filetype-suffix).
Default is `'my_cleaned_and_tokenized_data'`.
- __languages (list of str, optional):__ List of string codes for certain languages to filter for. Default is `None`.
See [Twitter language codes] for more info.
- __metadata (bool):__ Keep all available covariates or only the ones necessary for the package. Default is `False`.
- __min_tweet_len (int, optional):__ Re-filter for a minimal token number for each tweet after the cleaning process. Default is `None`.
- __spacy_model (str):__ Choose the desired *spacy model* for text tokenization. Non-default model installation tutorial
and an overview about the supported languages can be found at the [spacy website].
Default is the small *English* model called `'en_core_web_sm'`.

[spacy website]: https://spacy.io/usage/models

__Note:__ The user might decide the create a *Cleaner* object for other analytical purposes than for this package. In 
this case set `metadata=True` when instancing an object to get access to all the covariates! An overview can be found 
at [Covariates available after instancing a "Cleaner"-object].

[Covariates available after instancing a "Cleaner"-object]: covariates_cleaner_object.md

###Cleaner methods

__Method *saving:*__
```python
Cleaner.saving(save_path, type='pkl')
```
__Parameters:__
Saves the processed tweets as *pickle* or *csv* in batches of max. 300.000 tweets, each.

- __save_path (str):__ Path to where the resulting files shall be saved. 
- __type (str):__ File type to be saved. Choose between `'pkl'` and `'csv'`. Default is `'pkl'`.


##Class LDAAnalyzer
The class *LDAAnalyzer* provides the following functionalities:

- Loads cleaned data, from *Cleaner* object or path.<br>
- Pools tweets by hashtags using cosine similarity to create longer pseudo-documents for better LDA estimation.
- Creates n-gram tokens<br>
- Trains several LDA models on all tweets, decides for the n-best to be saved by coherence score
- Saves corpi, models and vocabularies.
- Calculates topic distributions and saves them.<br>
- Creates a dict containing the tweets sorted by day / month.<br>
- Appends the values of a selected topic distribution of each topic to each tweet as a new column.<br>
- Appends prevalence statistics about passed tokens to every tweet.<br>
- Saves an LDAAnalyzer-object.<br>
- Loads an LDAAnalyzer-object.<br>
- Provides a histogram of top-words for selected topics of a lda model.<br>
- Plots the mean topical prevalence over time for chosen topics.<br>
- Produces word clouds for topics of an lda model.<br>
- Scatter plot tweets from up to ten topics from the whole dataset or a time-series on a *matplotlib basemap*. The 
tweets are categorized by their individual maximum prevalence score for the passed topical prevalence column name.<br>

__Note:__ The *LDAAnalyzer* object acts a container for the results of the methods *hashtag_pooling* and
*lda_training*. After these methods are successfully applied, all other methods can be applied. 

```python
class LDAAnalyzer(load_path=None, raw_data=None, n_jobs=2, cs_threshold=0.5, output_type='all',
                  spacy_model='en_core_web_sm', ngram_min_count=10, ngram_threshold=300)
```
__Parameters:__

- __load_path (str, optional):__ Path containing the cleaned data. Define this argument or the `raw_data` argument,
but not both. Default is `None`.
- __raw_data (pandas DataFrame, optional):__ Pass the `self.raw_data` attribute from a previous instantiated 
*Cleaner*-object. Define this argument or `load_path`, but not both. Default is `None`.
- __n_jobs (int):__ Defines the number of CPU-cores to use for the hashtag-pooling and for LDA training. Default is `2`.
- __cs_threshold (float):__ Defines the value for the cosine-similarity threshold: 0 > cs > 1. It is advised to choose a
value between 0.5 and 0.9. Default is `0.5`.
- __output_type (str):__ Defines the type of tweets that are returned after the hashtag-pooling. Choose `'pool_hashtags'`
to return all hashtag-pools as well as all single tweets containing a hashtag. Choose `'pool'` to return only all
hashtag-pools. Choose `'all'` (or any other string) to return all hashtag-pools, all single tweets containing a
hashtag and all single tweets containing no hashtag (if any). Default is `'all'`.
- __spacy_model (str):__ Choose the desired *spacy model* for hashtag tokenization. Non-default model installation tutorial
and an overview about the supported languages can be found at the [spacy website].
Default is the small *English* model called `'en_core_web_sm'`.
- __ngram_min_count (int):__ Ignores all words and n-grams with total collected count lower than this value.
Default is `10`.
- __ngram_threshold (int):__ Represents a score threshold for forming the n-gram-phrases (higher means fewer phrases).
For details about the scores calculation, see [this]. Default is `300`.

[this]: https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.original_scorer

###LDAAnalyzer methods

__Method *hashtag_pooling:*__
```python
LDAAnalyzer.hashtag_pooling()
```
Pools tweets by hashtags using cosine similarity to create longer pseudo-documents for better LDA estimation and creates
n-gram tokens. The method applies an implementation of the pooling algorithm from [Mehrotra et al. 2013]. The method adds
the result as a new attribute to the *LDAAnalyzer* object itself (*self.lda_all_tweets_pooled*). It returns `None`.

__Attention!__ 
Since the method uses the python library *multiprocessing*, Windows user must apply the method using the following format:
```python
if __name__ == '__main__':
    LDAAnalyzer.hashtag_pooling()
```
The reason is that Windows does not provide [*os.fork* (scroll down to chapter "16.6.3.2. Windows")].

[Mehrotra et al. 2013]: https://dl.acm.org/doi/abs/10.1145/2484028.2484166 
[*os.fork* (scroll down to chapter "16.6.3.2. Windows")]: https://docs.python.org/2.7/library/multiprocessing.html#multiprocessing-programming

<br/><br/>
__Method *lda_training:*__
```python
LDAAnalyzer.lda_training(data_save_path, models_save_path, data_save_type='pkl', 
                         ngram_style='unigrams', filter_keep_n=15000, filter_no_below=10,
                         filter_no_above=0.85, 
                         topic_numbers_to_fit=[10, 20, 30, 50, 75, 100, 150],
                         n_saved_top_models=3)
```
This method trains several LDA models on all tweets and decides for the n-best to be kept by coherence score. Additionally,
it saves corpi, models and vocabularies to drive. Finally, it calculates the topic distributions for the chosen models
and attaches it as a new attribute to the *LDAAnalyzer* object itself (*self.lda_df_trained_tweets*). It returns `None`.

__Parameters:__

- __data_save_path (str):__ Path directing to where the topic distributions of the individual tweets shall be saved.
- __models_save_path (str):__ Path directing to where the trained LDA models, corpi and vocabularies shall be saved.
- __data_save_type (str):__ Decides in which file format the topic distributions of the individual tweets are saved.
Choose between `'pkl'` and `'csv'`. Default is `'pkl'`.
- __ngram_style (str):__ Defines the n-gram type. Choose between `unigrams` (default), `bigrams` and `trigrams`.
- __filter_keep_n (int):__ Token filtering before the LDA training regarding the DTM. Keep only the *n* most
occurring tokens. Default is `15.000`.
- __filter_no_below (int):__ Token filtering before the LDA training regarding the DTM. Keep only tokens occurring
at least *n* times. Default is `10`.
- __filter_no_above (float):__ Token filtering before the LDA training regarding the DTM. Keep only tokens that are
occurring in at least *m* percent of all documents (tweet pools and tweets). Value must be between 0 and 1. Default
is `0.85`. 
- __topic_numbers_to_fit (list of int):__ Each integer in this list is referring to the number of
topics chosen for a LDA model to be estimated. Default is `[10, 20, 30, 50, 75, 100, 150]`.
- __n_saved_top_models (int):__ keep only the *n* best scoring LDA models regarding topical coherence score.
Default is `3`.

<br/><br/>
__Method *time_series_producer:*__
```python
LDAAnalyzer.time_series_producer(ts_type='d')
```
This method creates a dict containing the processed tweets sorted by day or month. It attaches the time series as a new
attribute to the *LDAAnalyzer* object itself (*self.time_series*). It returns `None`.

__Parameters:__

- __ts_type (str):__ Defines the interval of the time series. Choose between (`'d'`)aily and (`'m'`)onthly. Default is `'d'`.

<br/><br/>
__Method *topic_prevalence_flattening:*__
```python
LDAAnalyzer.topic_prevalence_flattening(topic_prevalence_column_str, type='all',
                                        date_of_df_in_dict_str=None)
```
This method appends the values of a selected topic distribution of each topic to each tweet as a new column to the 
chosen attribute. It returns `None`.

__Parameters:__

- __topic_prevalence_column_str (str):__ String referring to the name of a topic distribution column of *token*, 
*bi-* or *tri*-type of `self.lda_df_trained_tweets`.
- __type (str):__ Defines on which *DataFrame* the method is applied. Choose between `'all'` (`self.lda_df_trained_tweets`)
and `'ts'` (a *time-series-dict* entry). Default is `'all'`.
- __date_of_df_in_dict_str (str, optional):__ Choose the *key*-string of the desired entry from the *time-series-dict*,
if `type='ts'` (one of the strings from `self.lda_df_trained_tweets['created_at']` in the form of `yy-mm-dd`). Default is
`None`.

<br/><br/>
__Method *word_count_prevalence:*__
```python
LDAAnalyzer.word_count_prevalence(searched_token_list, type='all', date_of_df_in_dict_str=None)
```
This method appends prevalence statistics about passed tokens to every tweet for chosen attribute. It returns `None`.

__Parameters:__

- __searched_token_list (list of str):__ List containing strings that are searched for.
- __type (str):__ Defines on which *DataFrame* the method is applied. Choose between `'all'` (`self.lda_df_trained_tweets`)
and `'ts'` (a *time-series-dict* entry). Default is `'all'`.
- __date_of_df_in_dict_str (str, optional):__ Choose the *key*-string of the desired entry from the *time-series-dict*,
if `type='ts'` (one of the strings from `self.lda_df_trained_tweets['created_at']` in the form of `yy-mm-dd`). Default
is `None`.

<br/><br/>
__Method *save_lda_analyzer_object:*__
```python
LDAAnalyzer.save_lda_analyzer_object(save_path, obj_name='my_LDAAnalyzer_Object.pkl')
```
Simple method to save a *LDAAnalyzer* object as *pkl* to drive. Returns the saved object.

__Parameters:__

- __save_path (str):__ Path to where the *LDAAnalyzer* object shall be saved.
- __obj_name (str):__ Name of the *pkl* object. Make sure the name ends with *.pkl*. Default is `'my_LDAAnalyzer_Object.pkl'`.

 <br/><br/>
__Method *load_lda_analyzer_object:*__
```python
LDAAnalyzer.load_lda_analyzer_object(load_path, obj_name)
```
Simple static method to load a *LDAAnalyzer* object. Returns the loaded object.

__Parameters:__

- __load_path (str):__ Path to where the *LDAAnalyzer* object is saved.
- __obj_name (str):__ Name of the *pkl* object to be loaded.

 <br/><br/>
__Method *plot_top_topics_from_lda:*__
```python
LDAAnalyzer.plot_top_topics_from_lda(lda_model_object, topics, num_top_words=10, save_path=None,
                                     save_name='my_topics_top_word_histogram')
```
This static method plots a histogram of top-words for selected topics of a lda model. It returns `None`.

__Parameters:__

- __lda_model_object (gensim model object):__ One of the *gensim* model objects saved in `self.lda_models`.
- __topics (list of int):__ List of integers corresponding to the designated topic numbers to be
plotted (i.e. [0,3] -> plots "Topic 0" and "Topic 3"). Maximum of 10 Topics at once!
- __num_top_words (int):__ Defines the number of words to be plotted for each topic. Default is `10`.
- __save_path (str, optional):__ Defines a save path to where the plot is saved as *PDF*. Default is `None`.
- __save_name (str, optional):__ Defines a name for the *PDF* file, if a `save_path` is chosen. Default
is `'my_topics_top_word_histogram'`.

 <br/><br/>
__Method *time_series_plot:*__
```python
LDAAnalyzer.time_series_plot(topical_prevalence_column_name, topics_to_plot, save_path=None,
                             save_name='my_mean_topical_prevalence_over_time_for_chosen_topics')
```
This method plots the mean topical prevalence over time for chosen topics. It returns `None`.

__Parameters:__

- __topical_prevalence_column_name (str):__ Defines the name of the column that shall be used for plotting from the
time series (one the strings from `self.lda_df_trained_tweets['created_at']` in the form of `yy-mm-dd`).
- __topics_to_plot (list of int):__ Defines a list of integers referring to the topics numbers to be plotted.
- __save_path (str, optional):__ Defines a save path to where the plot is saved as *PDF*. Default is `None`.
- __save_name (str, optional):__ Defines a name for the *PDF* file, if a `save_path` is chosen. Default
is `'my_mean_topical_prevalence_over_time_for_chosen_topics'`.
        
 <br/><br/>
__Method *wordcloud:*__
```python
LDAAnalyzer.wordcloud(lda_model_object_str, no_of_words, topics=None, save_path=None)
```       
This method plots word clouds for chosen topics of an lda model. It returns `None`.

__Parameters:__

- __lda_model_object_str (str):__ Defines a string referring to the name of one of the saved *LDA* models
(in `self.lda_models`).
- __no_of_words (int):__ Number of top words for each word cloud to plot.
- __topics (list of int, optional):__ Passes a list of integers referring to the topics to be plotted. If `None`
is passed, all topics are plotted. Default is `None`.
- __save_path (str, optional):__ Defines a save path to where the plots are saved as *PDF*. Default is `None`.

 <br/><br/>
__Method *loc_vis:*__
```python
LDAAnalyzer.loc_vis(topical_prevalence_column_name, topics_to_plot, type='all',
                    markersize=100, draw_lat_and_lon=False, date_of_df_in_dict_str=None, 
                    save_path=None, save_name='my_topics_spatial_visualization')
```
This method provides a scatter plot of tweets from up to ten topics from the whole dataset or a time-series on a 
*matplotlib basemap*. The tweets are categorized by their individual maximum prevalence score for the passed topical
prevalence column name. It returns `None`.

__Parameters:__

- __topical_prevalence_column_name (str):__ Defines the name of the topical prevalence column that shall be used for
plotting.
- __topics_to_plot (list of int):__ Defines a list of integers referring to the topics numbers to be plotted.
Maximum of ten topics.
- __type (str):__ Defines on which *DataFrame* the method is applied to. Choose between `'all'`
(`self.lda_df_trained_tweets`) and `'ts'` (a *time-series-dict* entry). Default is `'all'`.
- __markersize (int):__ Defines the size of the markers of the scatter plot. Default is `100`.
- __draw_lat_and_lon (bool):__ Decides, if latitudes and longitudes are provides as lines on the map.
Default is `False`.
- __date_of_df_in_dict_str (str, optional):__ Choose the *key*-string of the desired entry from the *time-series-dict*,
if `type='ts'` (one the strings from `self.lda_df_trained_tweets['created_at']` in the form of `yy-mm-dd`). Default 
is `None`.
- __save_path (str, optional):__ Defines a save path to where the plot is saved as *PDF*. Default is `None`.
- __save_name (str, optional):__ Defines a name for the *PDF* file, if a `save_path` is chosen. Default
is `'my_topics_spatial_visualization'`.
