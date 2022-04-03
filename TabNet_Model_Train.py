# -*- coding:utf-8-*-
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from def_fun import plot_history_loss, plot_history_mae_mse
import shap

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(10)

data = pd.read_excel('data/bistandard+soft_fill.xlsx')
data_x = data.iloc[:, :-1]
feat_importance = data_x.columns
data_y = data.iloc[:, -1]  # ！！！

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=True, random_state=10)
test_x, valid_x, test_y, valid_y = train_test_split(test_x, test_y, test_size=0.5, shuffle=True, random_state=10)

train_x = np.array(train_x)
valid_x = np.array(valid_x)
test_x = np.array(test_x)
train_y = np.array(train_y).reshape(-1, 1)
test_y = np.array(test_y).reshape(-1, 1)
valid_y = np.array(valid_y).reshape(-1, 1)

clf = TabNetRegressor(
    verbose=True,
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
    max_epochs=200,
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

plot_history_loss(clf.history)
plot_history_mae_mse(clf.history)


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
plot_data.plot(kind='bar')
plt.show()
