from gensim import corpora, models

def topic_analysis(texts:list):
    '''
    Topic分析を行う関数

    args:
        texts: テキストのリスト

    return:
        topic_list: Topicのリスト
    '''

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    hdp = models.HdpModel(corpus, dictionary, random_state=0)
    topic_list = hdp.print_topics(num_words=10)

    return topic_list
