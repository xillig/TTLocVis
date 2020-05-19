#Usage and Examples

This section provides a few coding examples applying the functionalities of the TTLocVis package. To get to know the
Package, it is advised to use the JSON *example_files* provided in the *github repository* as data input for the 
classes *Cleaner* and *LDAAnalyzer*, skipping the streaming at first. __Note that the provided example data is for
presentation purposes only and does not provide any meaningful results!__   

After [installation], import the necessary classes.
```python
from TTLocVis.module import TwitterStreamer
from TTLocVis.module import Cleaner
from TTLocVis.module import LDAAnalyzer
```
[installation]: index.md#Installation

##Streaming
The Twitter data can be accessed easily when a [Twitter developer account] has been created by just one statement:
```python
TwitterStreamer(auth_path=r'path-to-Twitter API Access.txt',
                save_path=r'path-to-save-location', languages=['en'],locations=[-125,25,-65,48])
```
In this case, english-language tweets from a bounding-box roughly covering the US are collected via the `languages` 
and `locations` argument. A tool to get bounding-box coordinates can be found [here].
 
__Note:__ The streamer collects data until an error from the API occurs or the user performs a *KeyboardInterrupt* 
(Ctrl+C) 

[here]: https://boundingbox.klokantech.com/
[Twitter developer account]: https://developer.twitter.com/en

##Cleaning
Cleaning and transforming the raw JSON files is as simple as the streaming:
```python
c = Cleaner(load_path=r'path-to-a-JSON-file-folder')
print(c.raw_data)
c.saving(r'path-to-where-the-cleaned-data-shall-be-saved')
```
The resulting data is printed and is saved in the object itself in the attribute `self.raw_data`.

##LDA Analysis and Visualization
Using the default spacy model for english tweets, the *LDAAnalyzer* can be instantiated like this, printing the 
loaded-in data:
```python
d = LDAAnalyzer(load_path=r'path-to-where-the-cleaned-data-was-saved')
print(d.data)
```
The hashtag pooling and preparing process can be applied by
```python
d.hashtag_pooling()
print(d.lda_all_tweets_pooled)
```
__Attention!__ 
Since the method uses the python library *multiprocessing*, Windows user must apply the method using the following format:
```python
if __name__ == '__main__':
    d.hashtag_pooling()
print(d.lda_all_tweets_pooled)
```
The reason is that Windows does not provide [*os.fork* (scroll down to chapter "16.6.3.2. Windows")].
[*os.fork* (scroll down to chapter "16.6.3.2. Windows")]: https://docs.python.org/2.7/library/multiprocessing.html#multiprocessing-programming

Next, the LDA training can be done:
```python
d.lda_training(data_save_path=r'path-to-where-the-topic-distributions-shall-be-saved',
               models_save_path=r'path-to-where-lda_models_corpi_and_vocabulary-shall-be-saved',
               ngram_style='bigrams', topic_numbers_to_fit=[3, 5], n_saved_top_models=2)
print(d.lda_df_trained_tweets)
print(d.lda_models)
```
The resulting object can be saved as a *pkl* file: 
```python
d.save_lda_analyzer_object(save_path=r'path-to-where-the-lda_analyzer_object-shall-be-saved')
```
It can be reloaded as easy as it is saved:
```python
q = LDAAnalyzer.load_lda_analyzer_object(load_path=r'path-to-where-the-lda_analyzer_
                                                   object-was-saved',
                                         obj_name='my_LDAAnalyzer_Object.pkl') 
```
A time series of the data can be produced using the corresponding method:
```python
q.time_series_producer(ts_type='d')
print(q.time_series)
```
The topic prevalence for each topic as an individual column of a chosen model can be appended to the main dataframe
(all dates) using the *topic_prevalence_flattening* method:
```python
q.topic_prevalence_flattening('lda_5_topics_bigrams')
print(q.lda_df_trained_tweets)
```
The same is possible for certain dates explicitly using the produced time series:
```python
q.topic_prevalence_flattening('lda_5_topics_bigrams', type='ts',
                              date_of_df_in_dict_str='19-10-26')
print(q.time_series)
```
Appending information about certain tokens is possible for the main dataframe as well as for certain dates of the time
series:
```python
q.word_count_prevalence(['open','halloween'])  # main data frame
print(q.lda_df_trained_tweets)
q.word_count_prevalence(['open','halloween'], type='ts', 
                        date_of_df_in_dict_str='19-10-26')  # certain time series
print(q.time_series)
```
###Visualisation
This method plots a histogram of top-words for selected topics of a lda model:
```python
LDAAnalyzer.plot_top_topics_from_lda(q.lda_models['lda_5_topics_bigrams'], topics=[1,3],
                                     save_path=r'path-to-where-the-plot-shall-be-saved-as-pdf')
```
This method plots the mean topical prevalence over time for chosen topics:
```python
q.time_series_plot(topical_prevalence_column_name='lda_5_topics_bigrams',topics_to_plot=[0,2],
                   save_path=r'path-to-where-the-plot-shall-be-saved-as-pdf')      
```
It is possible to produce word clouds for every topic:
```python
q.wordcloud(lda_model_object_str='lda_5_topics_bigrams', no_of_words=20,
            topics=[0,3], save_path=r'path-to-where-the-plot-shall-be-saved-as-pdf')
```
Finally, this method provides a scatter plot of tweets from up to ten topics from the whole dataset or a time-series 
on a *matplotlib basemap*. The tweets are categorized by their individual maximum prevalence score for the passed topical
prevalence column name.
__Note:__ This method is unfortunately unavailable for users of *iOS*.
```python
q.loc_vis(topical_prevalence_column_name='lda_5_topics_bigrams',topics_to_plot=[0,1,2,3,4],
          type='ts', date_of_df_in_dict_str='19-10-26')
```









