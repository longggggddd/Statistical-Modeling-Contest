# -*- coding:utf-8-*-
from transformers import BertTokenizer, BertModel, pipeline, BertConfig, BertForSequenceClassification
from transformers import AdamW
import torch
from sklearn.model_selection import StratifiedKFold
from transformers import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_history_loss(history):
    loss = history['loss']
    plt.figure(figsize=(12, 5))
    plot_data = pd.DataFrame(data={
        "loss": loss,

    })
    sns.lineplot(data=plot_data)
    plt.title('Loss in Validation')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.legend()
    plt.savefig('picture/loss.png')
    plt.show()


def plot_history_mae_mse(history):
    val_0_rmse = history['val_0_rmse']
    val_0_mae = history['val_0_mae']
    plt.figure(figsize=(12, 5))
    plot_data = pd.DataFrame(data={

        "val-rmse": val_0_rmse,
        'val-mae': val_0_mae
    })  # 添加注释
    sns.lineplot(data=plot_data)
    plt.title('MSE and MAE in Validation')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.legend()
    plt.savefig('picture/mae_mse.png')
    plt.show()
