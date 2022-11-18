
# ## 1.预处理

# In[3]:

import os
import pandas as pd
import re
import jieba
import jieba.posseg as psg

# In[4]:

output_path = 'C:/Users/jia/Desktop/lda-描述/result'
file_path = 'C:/Users/jia/Desktop/lda-描述/data'
os.chdir(file_path)
data=pd.read_csv("data_describe.csv")#content type
os.chdir(output_path)
stop_file = "C:/Users/jia/Desktop/lda-描述/stop_dic/stopwords.txt"


# In[35]:


def chinese_word_cut(mytext):
    #jieba.load_userdict(dic_file)
    jieba.initialize()
    try:
        stopword_list = open(stop_file,encoding ='utf-8')
    except:
        stopword_list = []
        print("error in stop_file")
    stop_list = []
    flag_list = ['n','nz','vn']
    for line in stopword_list:
        line = re.sub(u'\n|\\r', '', line)
        stop_list.append(line)
    
    word_list = []
    #jieba分词
    seg_list = psg.cut(mytext)
    for seg_word in seg_list:
        word = re.sub(u'[^\u4e00-\u9fa5]','',seg_word.word)
        #word = seg_word.word  #如果想要分析英语文本，注释这行代码，启动下行代码
        find = 0
        for stop_word in stop_list:
            if stop_word == word or len(word)<2:     #this word is stopword
                    find = 1
                    break
        if find == 0 and seg_word.flag in flag_list:
            word_list.append(word)      
    return (" ").join(word_list)


# In[36]:


data["描述_cutted"] = data.描述.apply(chinese_word_cut)


# ## 2.LDA分析

# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[38]:


def print_top_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword


# In[39]:


n_features = 500 #提取1000个特征词语
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df = 0.5,
                                min_df = 10)
tf = tf_vectorizer.fit_transform(data.描述_cutted)


# In[40]:


n_topics = 9
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='batch',
                                learning_offset=50,
#                                 doc_topic_prior=0.1,
#                                 topic_word_prior=0.01,
                               random_state=0)
lda.fit(tf)


# ### 2.1输出每个主题对应词语 

# In[11]:


n_top_words = 20
tf_feature_names = tf_vectorizer.get_feature_names()
topic_word = print_top_words(lda, tf_feature_names, n_top_words)


# ### 2.2输出每篇文章对应主题 

# In[12]:


import numpy as np


# In[13]:


topics=lda.transform(tf)


# In[28]:


topic = []
for t in topics:
    topic.append("Topic #"+str(list(t).index(np.max(t))))
data['概率最大的主题序号']=topic
data['每个主题对应概率']=list(topics)
data.to_excel("data_topic.xlsx",index=False)


# ### 2.3可视化 

# In[29]:


import pyLDAvis
import pyLDAvis.sklearn


# In[31]:

pic = pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)
#pyLDAvis.display(pic)
pyLDAvis.save_html(pic, 'lda_pass'+str(n_topics)+'.html')
#pyLDAvis.display(pic)



# ### 2.4困惑度 

# In[32]:


import matplotlib.pyplot as plt


# In[41]:


plexs = []
scores = []
n_max_topics = 16
for i in range(1,n_max_topics):
    print(i)
    lda = LatentDirichletAllocation(n_components=i, max_iter=50,
                                    learning_method='batch',
                                    learning_offset=50,random_state=0)
    lda.fit(tf)
    plexs.append(lda.perplexity(tf))
    scores.append(lda.score(tf))


# In[42]:


n_t=10#区间最右侧的值。注意：不能大于n_max_topics
x=list(range(1,n_t+1))
plt.plot(x,plexs[0:n_t])
plt.xlabel("number of topics")
plt.ylabel("perplexity")
plt.show()


# In[ ]:




