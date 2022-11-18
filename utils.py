# -*- coding:utf-8-*-
import numpy as np
from transformers import BertTokenizer, BertModel, pipeline, BertConfig, BertForSequenceClassification
from transformers import AdamW
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jieba_fast as jieba
from wordcloud import WordCloud
plt.style.use('ggplot')


def kfold_stats_feature(train, feats, y, k, seed):
    '''
    Target-Encode
    '''

    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train[y])):
        train.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in feats:
        nums_columns = [y]
        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            kfold_features.append(colname)
            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train[y])):
                tmp_trn = train.iloc[trn_idx]
                order_label = tmp_trn.groupby([feat])[f].mean()
                tmp = train.loc[train.fold == fold_, [feat]]
                train.loc[train.fold == fold_, colname] = tmp[feat].map(order_label)
                # fillna
                global_mean = train[f].mean()
                train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(global_mean)
            train[colname] = train[colname].astype(float)

    del train['fold']
    return train


def mean_score(lstsentence):
    logging.set_verbosity_warning()
    MODEL_PATH = "chinese-bert_chinese_wwm_pytorch"
    a = pipeline('sentiment-analysis', model=MODEL_PATH)
    result = a(lstsentence)
    return float(result[0]['score'])


def plot_history_loss(history,name):
    loss = history['loss']
    plt.figure(figsize=(12, 5))
    plot_data = pd.DataFrame(data={
        "Loss": loss,

    })
    sns.lineplot(data=plot_data)
    plt.title('Loss in Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'picture/{name}-loss.png')
    plt.show()


def plot_history_mae_mse(history, name='pre'):
    val_0_rmse = history['val_0_rmse']
    val_0_mae = history['val_0_mae']
    plt.figure(figsize=(12, 5))
    plot_data = pd.DataFrame(data={

        "RMSE": val_0_rmse,
        'MAE': val_0_mae
    })  # 添加注释
    sns.lineplot(data=plot_data)
    plt.title('RMSE and MAE in Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'picture/{name}-mae-mse.png')
    plt.show()


def term_f(n, text):
    with open('data/stopwords.txt', 'r', encoding="utf-8") as f:
        lines = f.readlines()
        f.close()
    stopwords = []
    for l in lines:
        stopwords.append(l.strip())

    list_words = jieba.lcut(text)
    list_words1 = []
    for i in list_words:
        if i != " " and i not in stopwords:
            list_words1.append(i)

    # print(list_words1)
    dict1 = {}
    for ii in list_words1:
        dict1[ii] = list_words1.count(ii)
    for key in list(dict1):  # 注意字典在遍历时候不可以修改里面的内容！！！遍历修改需要转换成列表！！
        if len(key) == 1:
            del dict1[key]
    s = list(set(list(dict1.values())))
    s.sort()
    s.reverse()
    va = s[0:]
    ln = []
    for ii in va:
        for i in list(dict1.items()):
            if ii == i[1]:
                ln.append(i)

    print("本文出现词语频数如下：")
    for n in ln[:n]:
        print(("{:<5} {:>10}次".format(n[0], n[1])))


def tan(address):
    words = jieba.lcut(address)
    for i in words:
        if len(i) == 1:
            words.remove(i)  # 字符串长度为1的不予以输出
    stopwords = open('data/stopwords.txt', encoding='utf-8').read().split()
    text_list = [item for item in words if item != ' ' and item not in stopwords]
    new = " ".join(text_list)
    return new


def paint(n, text, name='describe'):
    # mask = np.array(Image.open("heart.jpg"))

    text = tan(text)
    wordcloud = WordCloud(background_color="white", width=750, height=630, max_words=n,
                          # mask=mask,
                          # 做词云处理的时候mask后边接的是处理成数组之后的图片
                          contour_width=3, contour_color="green",
                          font_path="data/微软雅黑.ttf"
                          ).generate(
        text)  # 必须是字符串 中间用空白间隔的字符串
    # wordcloud.to_file(r"C:/Users/123/Desktop/Python学习笔记/作业文档/词云结果图/词云结果图.png")  # 做好的图片放到一个新的文件之中
    plt.imshow(wordcloud)  # 将图片文件显示在坐标图之上
    plt.axis("off")  # 无坐标轴
    plt.savefig(f'picture/{name}.png')
    plt.show()  # 显示图片
