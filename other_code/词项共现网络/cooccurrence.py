
# coding: utf-8




import jieba
import os
import re
import numpy as np
import jieba_fast.posseg as psg
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



def get_stop_dict(file):
    content = open(file, encoding="utf-8")
    word_list = []
    for c in content:
        c = re.sub('\n|\r', '', c)
        word_list.append(c)
    return word_list



def get_data(path):
    t = open(path, encoding="utf-8")
    data = t.read()
    t.close()
    return data




def get_wordlist(text, maxn, synonym_words, stop_words):
    synonym_origin = list(synonym_words['origin'])
    synonym_new = list(synonym_words['new'])
    flag_list = ['n', 'nz', 'vn']  # a,形容词，v,形容词
    counts = {}

    text_seg = psg.cut(text)
    for word_flag in text_seg:
        # word = re.sub("[^\u4e00-\u9fa5]","",word_flag.word)
        word = word_flag.word
        if word_flag.flag in flag_list and len(word) > 1 and word not in stop_words:
            if word in synonym_origin:
                index = synonym_origin.index(word)
                word = synonym_new[index]
            counts[word] = counts.get(word, 0) + 1

    words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    words = list(dict(words).keys())[0:maxn]

    return words




def get_t_seg(topwords, text, synonym_words, stop_words):
    synonym_origin = list(synonym_words['origin'])
    synonym_new = list(synonym_words['new'])
    flag_list = ['n', 'nz', 'vn']  # a,形容词，v,形容词

    text_lines_seg = []
    text_lines = text.split("\n")
    for line in text_lines:
        t_seg = []
        text_seg = psg.cut(line)
        for word_flag in text_seg:
            # word = re.sub("[^\u4e00-\u9fa5]","",word_flag.word)
            word = word_flag.word
            if word_flag.flag in flag_list and len(word) > 1 and word not in stop_words:
                if word in synonym_origin:
                    word = synonym_new[synonym_origin.index(word)]
                if word in topwords:
                    t_seg.append(word)
        t_seg = list(set(t_seg))
        text_lines_seg.append(t_seg)
    return text_lines_seg




def get_comatrix(text_lines_seg):
    comatrix = pd.DataFrame(np.zeros([len(topwords), len(topwords)]), columns=topwords, index=topwords)
    for t_seg in text_lines_seg:
        for i in range(len(t_seg) - 1):
            for j in range(i + 1, len(t_seg)):
                comatrix.loc[t_seg[i], t_seg[j]] += 1
    for k in range(len(comatrix)):
        comatrix.iloc[k, k] = 0
    return comatrix


# In[7]:


def get_net(co_matrix, topwords):
    G = nx.Graph()
    w_lst = []
    for i in range(len(topwords) - 1):
        word = topwords[i]
        for j in range(i + 1, len(topwords)):
            w = 0
            word2 = topwords[j]
            w = co_matrix.loc[word][word2] + co_matrix.loc[word2][word]
            if w > 0:
                G.add_edge(word, word2, weight=w)
                w_lst.append(w)
    print(w_lst)
    theiaod = np.mean(w_lst)
    print(theiaod)
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > theiaod]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= theiaod]
    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=200)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=0.5, alpha=0.5, edge_color="b", style="dashed"
    )

    # labels
    nx.draw_networkx_labels(G, pos, font_size=7, font_family="sans-serif")

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    return G


# In[8]:


# 文件路径
dic_file = "stop_dic/dict.txt"
stop_file = "stop_dic/stopwords.txt"
data_path = "postprocess_data/reviews_data.txt" # postprocess_data/reviews_data.txt
synonym_file = "stop_dic/synonym_list.xlsx"

# In[9]:


# 读取文件
data = get_data(data_path)
stop_words = get_stop_dict(stop_file)
jieba.load_userdict(dic_file)
synonym_words = pd.read_excel(synonym_file)




# 数据处理
n_topwords = 30
topwords = get_wordlist(data, n_topwords, synonym_words, stop_words)
t_segs = get_t_seg(topwords, data, synonym_words, stop_words)

# In[11]:


co_matrix = get_comatrix(t_segs)
co_net = get_net(co_matrix, topwords)

print(co_net)

