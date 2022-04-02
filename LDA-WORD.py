# -*- coding:utf-8-*-
import re
from langconv import Converter
import pkuseg
import pandas as pd
import jieba_fast as jieba
from sklearn.feature_extraction.text import CountVectorizer
import lda
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def cht_2_chs(line):
    line = Converter('zh-hans').convert(line)
    line.encode('utf-8')
    return line


def preprocess(text, stopwords):
    token = "[0-9\s+\.\!\/_,$%^*()?;；：" \
            "【】+\"\'\[\]\\]+" \
            "|[+——！，" \
            ";:。？《》、~@#￥%……&*（）“”.=-]+"
    seg = pkuseg.pkuseg()
    text1 = re.sub('&nbsp', ' ', text)
    str_no_punctuation = re.sub(token, ' ', text1)
    text_list = jieba.lcut(str_no_punctuation)
    text_list = [item for item in text_list if item != ' ' and item not in stopwords]
    return cht_2_chs(' '.join(text_list))


def plot_topic(doc_topic):
    f, ax = plt.subplots(figsize=(10, 4))
    cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(doc_topic, cmap=cmap, linewidths=0.05, ax=ax)
    ax.set_title('proportion per topic in every homestay')
    ax.set_xlabel('topic')
    ax.set_ylabel('homestay')
    plt.show()
    f.savefig('picture/topic_heatmap.png', bbox_inches='tight')


def lda_train(weight, vectorizer, N_TOPICS, n_top_words):
    model = lda.LDA(n_topics=N_TOPICS, n_iter=500, random_state=1)
    model.fit(weight)

    doc_num = len(weight)
    topic_word = model.topic_word_
    vocab = vectorizer.get_feature_names()
    titles = ["民宿{}".format(i) for i in range(1, doc_num + 1)]

    n_top_words = n_top_words
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    doc_topic = model.doc_topic_
    # print(doc_topic, type(doc_topic))
    plot_topic(doc_topic)
    # for i in range(doc_num):
    # print("{} (top topic: {})".format(titles[i], np.argsort(doc_topic[i])[:-4:-1]))


def main(num_of_homestay, num_of_topic, num_of_topic_words):
    data_good = pd.read_csv('数据/original_data.csv', encoding='ANSI')
    with open('数据/stopwords.txt', 'r', encoding="utf-8") as f:
        lines = f.readlines()
        f.close()
    stopwords = []
    for l in lines:
        stopwords.append(l.strip())
    original = data_good['描述'][:num_of_homestay]
    corpus = [preprocess(i, stopwords) for i in original]
    # corpus = [' '.join(corpus)]
    # print(corpus)
    cntVector = CountVectorizer(stop_words=stopwords)
    cntTf = cntVector.fit_transform(corpus)
    N_TOPICS = num_of_topic
    N_TOP_WORDS = num_of_topic_words
    lda_train(cntTf.toarray(), cntVector, N_TOPICS, N_TOP_WORDS)


if __name__ == '__main__':
    num_of_homestay = 20  # 前N个民宿，不能用全部，太多了，我试了最多50个最好
    N_TOPICS = 10  # 提取N个主题
    N_TOP_WORDS = 10  # 每个主题提取N个词
    main(num_of_homestay, N_TOPICS, N_TOP_WORDS)
