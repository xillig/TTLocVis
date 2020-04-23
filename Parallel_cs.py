#used in the script "LDA Preparation". STORE IT IN THE SCRIPTS FOLDER!
def parallel(pooled_to_vectorize, cs_threshold, len_pooled, vectorizer_fit, single_tweet_vectorizer_fit_unpooled):

    from sklearn.metrics.pairwise import cosine_similarity
    #pooled_to_vectorize: nested list of all tokens, not separated by comma anymore but only by whitespace.
    #cs_threshold: cosine similarity threshold. see markdown explanation about "Hashtag Labeling Algorithm" in script "LDA Preparation".
    #len_pooled: length of all hashtag-pooled tweets.
    #vectorizer_fit: fitted TF-IDF values for all tweets
    #single_tweet_vectorizer_fit_unpooled: list containing fitted TF-IDF values of the single (unpooled) tweets. no. of entries is defined by no. of workers
    indices_unlabeled = []
    value_of_cs = []
    indices_to_append_atp = []


    #check all hashtag pool vs. one single tweet at a time at any worker.
    cs = cosine_similarity(vectorizer_fit[:len_pooled], single_tweet_vectorizer_fit_unpooled[0]).flatten()

    #gets the index with the highest cs score for a pooled hashtag.
    most_similar_tweets_index = cs.argsort()[:-2:-1]

    if cs[most_similar_tweets_index] > cs_threshold: #cs min. default: 0.5
        indices_unlabeled.append(single_tweet_vectorizer_fit_unpooled[1]) #index number of unlabeled tweet in 'all_tweets_pool'
        value_of_cs.append(cs[most_similar_tweets_index][0]) #corresponding cs value
        indices_to_append_atp.append(pooled_to_vectorize.index[most_similar_tweets_index][0]) #index number of hashtag pool in 'all_tweets_pool', where the single tweet shall be appended.

        return indices_unlabeled, value_of_cs, indices_to_append_atp

    else:
        return #return 'None' if cs of single tweet couldn't pass the threshold
