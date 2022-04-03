# -*- coding:utf-8-*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import seaborn as sns
import shap

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(10)

data = pd.read_excel('data/bistandard+soft_fill.xlsx')
data_x = data.iloc[:, :-1]
feat_importance = data_x.columns
data_y = data.iloc[:, -1]  # ！！！

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=True, random_state=10)

clf_list = [XGBRegressor(), LinearRegression(), DecisionTreeRegressor(), MLPRegressor(hidden_layer_sizes=(50, 50))]
clf_name = ['XGB', 'LR', 'DT', 'MLP']
for i, clf in enumerate(clf_list):
    print('=' * 30, clf_name[i], '=' * 30)
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)

    mse = mean_squared_error(test_y, pred_y)
    mae = mean_absolute_error(test_y, pred_y)
    r2 = r2_score(test_y, pred_y)

    print('RMSE:', np.sqrt(mse))
    print('MAE:', mae)
    print('R2:', r2)

# 下面计算特征重要性（shap使用）
# shap.initjs()
# explainer = shap.TreeExplainer(clf)  # 什么模型用什么解释器，有专门的深度学习解释器
# shap_values = explainer.shap_values(train_x)
# shap.force_plot(explainer.expected_value, shap_values[0, :], train_x[0, :], matplotlib=True)
# # shap.force_plot(explainer.expected_value, shap_values, train_x,matplotlib=True)
# shap.summary_plot(shap_values, train_x)
