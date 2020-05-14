#test the package!
#import TTLocVis.module
from TTLocVis.module import TwitterStreamer
from TTLocVis.module import Cleaner
from TTLocVis.module import LDAAnalyzer

# Apply the whole class!
# Data Scraping
#tw_streamer = TwitterStreamer(r'C:\Users\gilli\OneDrive\Dokumente\Uni\Masterarbeit\Wichtige Informationen\Twitter API Access.txt',
#                              save_path=r'C:\Users\gilli\OneDrive\Desktop', languages=['en'],locations=[-125,25,-65,48],
#                              hashtag=False)

# Data Cleaning
#c = Cleaner(load_path=r'C:\Users\gilli\OneDrive\Desktop')
# print(c.raw_data)
#c.saving(r'C:\Users\gilli\OneDrive\Desktop')

# LDA Analysis
if __name__ == '__main__':  # Mandatory for windows! see: https://stackoverflow.com/questions/58323993/passing-a-class-to-multiprocessing-pool-in-python-on-windows
    #d = LDAAnalyzer(load_path=r'C:\Users\gilli\OneDrive\Desktop')
    # print(type(d.data))
    #d.hashtag_pooling()
    #d.lda_training(data_save_path=r'C:\Users\gilli\OneDrive\Desktop\test',
    #               models_save_path=r'C:\Users\gilli\OneDrive\Desktop\test',
    #               ngram_style='bigrams', topic_numbers_to_fit=[3, 5], n_saved_top_models=2)
    #d.save_lda_analyzer_object(save_path=r'C:\Users\gilli\OneDrive\Desktop\test')
    q = LDAAnalyzer.load_lda_analyzer_object(load_path=r'C:\Users\gilli\OneDrive\Desktop\test', obj_name='my_LDAAnalyzer_Object.pkl')
    q.time_series_producer()
    q.topic_prevalence_flattening('lda_5_topics_bigrams')
    q.topic_prevalence_flattening('lda_5_topics_bigrams', type='ts',date_of_df_in_dict_str='19-10-26')
    q.word_count_prevalence(['open','hari'], type='ts',date_of_df_in_dict_str='19-10-26')
    LDAAnalyzer.plot_top_topics_from_lda(q.lda_models['lda_5_topics_bigrams'], topics=[1,3], save_path=r'C:\Users\gilli\OneDrive\Desktop\test')
    ################
    # Do change the date and variate values to test " time_series_plot"
    #q.time_series['20-04-18'] = q.time_series['20-04-17'].copy()
    #q.time_series['20-04-18'].loc[:,'lda_5_topics_bigrams'] = q.time_series['20-04-18'].loc[:,'lda_5_topics_bigrams'].apply(lambda x: [i-0.1 for i in x])
    #q.time_series['20-04-19'] = q.time_series['20-04-17'].copy()
    #q.time_series['20-04-19'].loc[:,'lda_5_topics_bigrams'] = q.time_series['20-04-19'].loc[:,'lda_5_topics_bigrams'].apply(lambda x: [i-0.2 for i in x])
    #print(q.time_series['20-04-17'].loc[:10,'lda_5_topics_bigrams'])
    #print(q.time_series['20-04-18'].loc[:10, 'lda_5_topics_bigrams'])
    #print(q.time_series)
    ############################
    q.time_series_plot(topical_prevalence_column_name='lda_5_topics_bigrams', topics_to_plot=[0,2], save_path=r'C:\Users\gilli\OneDrive\Desktop\test')
    q.wordcloud(lda_model_object_str='lda_5_topics_bigrams', no_of_words=20, topics=[0,3], save_path=r'C:\Users\gilli\OneDrive\Desktop\test')
    q.loc_vis(topical_prevalence_column_name='lda_5_topics_bigrams',topics_to_plot=[0,1,2,3,4], type='ts', date_of_df_in_dict_str='19-10-26')
