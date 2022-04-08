# -*- coding:utf-8-*-
# import transformers
# import tensorflow
# import keras
# import torch
# import sklearn
# print(sklearn.__version__)
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# print('transformers版本：', transformers.__version__)
# print('tensorflow版本：', tensorflow.__version__)
# print('keras版本：', keras.__version__)
# print('torch版本：', torch.__version__)

# data = np.random.randint(10, 100, (4, 4))
# print(data)
#
# scl = StandardScaler()
# data = scl.fit_transform(data)
# print(data)
# print(StandardScaler.__name__)

# from TabNet_Model_Train_Supervise import super
# from Tabnet_pre_train import sesuper
#
# a, b = super()
# c, d = sesuper()
#
# print(d)
# print(b)
# print(a)
# print(c)
#
# name = ['With-pretrain', 'Supervise']
# value = [c, a]
#
# plot_data = pd.DataFrame(data={
#     # "x": x,
#     "With-Pretrain": d,
#     "Supervise": b
# })
#
# plt.style.use('ggplot')
# plt.figure()
# sns.lineplot(data=plot_data)
# plt.legend()
# plt.savefig('picture/pre-train-and-supervise-loss.png')
# plt.show()
#
# plot_data = pd.DataFrame(data={
#     'name': name,
#     'R2': value
# })
# plot_data = plot_data.set_index(keys='name', drop=True)
# plot_data.plot(kind='bar')
# plt.savefig('picture/pre-train-and-supervise-r2.png')
# plt.show()
b = ['11', 'ss']
c = [1, 2]
zi = zip(b,c)
di=dict(zi)
print(di)
a = {'a': 1, 'b': 2}
a = sorted(a.items(), key=lambda x: x[1], reverse=True)
print(a)
