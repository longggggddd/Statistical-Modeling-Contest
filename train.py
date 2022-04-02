# -*- coding:utf-8-*-
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from def_fun import plot_history
import shap

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(10)

data = pd.read_excel('data/bistandard+soft_fill.xlsx')


data_x = data.iloc[:, :-1]
feat_importance = data_x.columns
data_y = data.iloc[:, -1]  # ！！！


train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=True, random_state=10)

# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)

# scl = [StandardScaler(), MinMaxScaler()][0]
# train_x = scl.fit_transform(train_x)
# test_x = scl.fit_transform(test_x)

clf = [XGBRegressor(), LinearRegression(), DecisionTreeRegressor(), MLPRegressor(hidden_layer_sizes=(50, 50))][0]
clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)

mse = mean_squared_error(test_y, pred_y)
mae = mean_absolute_error(test_y, pred_y)
r2 = r2_score(test_y, pred_y)

print(np.sqrt(mse))
print(mae)
print(r2)

test_x, valid_x, test_y, valid_y = train_test_split(test_x, test_y, test_size=0.5, shuffle=True, random_state=10)

train_x = np.array(train_x)
valid_x = np.array(valid_x)
test_x = np.array(test_x)
train_y = np.array(train_y).reshape(-1, 1)
test_y = np.array(test_y).reshape(-1, 1)
valid_y = np.array(valid_y).reshape(-1, 1)

# scl = [StandardScaler(), MinMaxScaler()][0]
# train_x = scl.fit_transform(train_x)
# valid_x = scl.fit_transform(valid_x)
# test_x = scl.fit_transform(test_x)

# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)


clf = TabNetRegressor(
    verbose=False,
    seed=10,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size": 10,  # how to use learning rate scheduler
                      "gamma": 0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR
)
clf.fit(
    train_x, train_y,
    batch_size=256,
    max_epochs=256,
    eval_set=[(valid_x, valid_y)],
    eval_metric=['rmse', 'mae'],
    patience=0
)

pred_y = clf.predict(test_x)

pred_y = pred_y.reshape(1, -1)[0]
test_y = test_y.reshape(1, -1)[0]

mse = mean_squared_error(test_y, pred_y)
mae = mean_absolute_error(test_y, pred_y)
r2 = r2_score(test_y, pred_y)
importance = clf.feature_importances_

print('RMSE:', np.sqrt(mse))
print('MAE:', mae)
print('R2:', r2)

plot_history(clf.history)
plot_data = pd.DataFrame(data={
    # "x": x,
    "true": test_y,
    "pred": pred_y
})

plt.style.use('ggplot')
plt.figure()
sns.lineplot(data=plot_data)
plt.legend()
plt.savefig('picture/true_pred.png')
plt.show()

plot_data = pd.DataFrame(data={
    # "x": x,
    "true": test_y,
    "pred": pred_y
})

plt.style.use('ggplot')
plt.figure()
sns.lineplot(data=plot_data)
plt.legend()
plt.savefig('picture/true_pred.png')
plt.show()

plot_data = pd.DataFrame(data={
    'feat_name': feat_importance,
    'importance': importance
})
plot_data = plot_data.set_index(keys='feat_name', drop=True)
print(plot_data)
plot_data.plot(kind='bar')
plt.show()
# 下面计算特征重要性（shap使用）
# shap.initjs()
# explainer = shap.TreeExplainer(clf)  # 什么模型用什么解释器，有专门的深度学习解释器
# shap_values = explainer.shap_values(train_x)
# shap.force_plot(explainer.expected_value, shap_values[0, :], train_x[0, :], matplotlib=True)
# # shap.force_plot(explainer.expected_value, shap_values, train_x,matplotlib=True)
# shap.summary_plot(shap_values, train_x)
