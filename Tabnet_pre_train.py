# -*- coding:utf-8-*-
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from def_fun import plot_history_loss, plot_history_mae_mse
import shap
from yellowbrick.model_selection import FeatureImportances
from yellowbrick.regressor import ResidualsPlot
import plotly.io as pio

pio.renderers.default = 'browser'

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

print(train_x.shape)
print(valid_x.shape)
print(test_x.shape)

# tabnet_params = {"cat_idxs":cat_idxs,
#                  "cat_dims":cat_dims,
#                  "cat_emb_dim":1,
#                  "optimizer_fn":torch.optim.Adam,
#                  "optimizer_params":dict(lr=2e-2),
#                  "scheduler_params":{"step_size":50, # how to use learning rate scheduler
#                                  "gamma":0.9},
#                  "scheduler_fn":torch.optim.lr_scheduler.StepLR,
#                  "mask_type":'entmax' # "sparsemax"
#                 }

# pre_clf = TabNetPretrainer(
#     verbose=True,
#     seed=10,
#     optimizer_fn=torch.optim.Adam,
#     optimizer_params=dict(lr=2e-2),
#     scheduler_params={"step_size": 10,  # how to use learning rate scheduler
#                       "gamma": 0.9},
#     scheduler_fn=torch.optim.lr_scheduler.StepLR,
#     # n_shared_decoder=1,  # nb shared glu for decoding
#     # n_indep_decoder=1,  # nb independent glu for decoding
#     # mask_type='entmax'  # "sparsemax"
# )
# pre_clf.fit(
#     X_train=train_x,
#     eval_set=[valid_x],
#     max_epochs=200,
#     batch_size=256,
#     num_workers=0,
#     drop_last=False,
#     pretraining_ratio=0.8,
#     patience=0
# )
#
# # Make reconstruction from a dataset
# reconstructed_X, embedded_X = pre_clf.predict(valid_x)
# assert (reconstructed_X.shape == embedded_X.shape)
# unsupervised_explain_matrix, unsupervised_masks = pre_clf.explain(valid_x)
# pre_clf.save_model('pretrain/test_pretrain')

loaded_pretrain = TabNetPretrainer()
loaded_pretrain.load_model('pretrain/test_pretrain.zip')
clf = TabNetRegressor(optimizer_fn=torch.optim.Adam,
                      optimizer_params=dict(lr=2e-2),
                      scheduler_params={"step_size": 10,  # how to use learning rate scheduler
                                        "gamma": 0.9},
                      scheduler_fn=torch.optim.lr_scheduler.StepLR,
                      mask_type='sparsemax'  # This will be overwritten if using pretrain model
                      )
clf.fit(
    train_x, train_y,
    batch_size=256,
    max_epochs=200,
    eval_set=[(valid_x, valid_y)],
    eval_metric=['rmse', 'mae'],
    patience=0,
    from_unsupervised=loaded_pretrain)

pred_y = clf.predict(test_x)
pred_y = pred_y.reshape(1, -1)[0]
test_y = test_y.reshape(1, -1)[0]

mse = mean_squared_error(test_y, pred_y)
mae = mean_absolute_error(test_y, pred_y)
r2 = r2_score(test_y, pred_y)
importance = clf.feature_importances_
# print(clf.network)

# 局部重要性
explain_matrix, masks = clf.explain(test_x)
fig, axs = plt.subplots(1, 3, figsize=(20, 20))

for i in range(3):  # 代表展示前3个MASK
    axs[i].imshow(masks[i][:50])  # 50 代表前50个样本
    axs[i].set_title(f"mask {i}")

print('MSE:', mse)
print('MAE:', mae)
print('R2:', r2)

plot_history_loss(clf.history, 'pre')
plot_history_mae_mse(clf.history, 'pre')

plot_data = pd.DataFrame(data={
    # "x": x,
    "True": test_y,
    "Pred": pred_y
})

plt.style.use('ggplot')
plt.figure()
sns.lineplot(data=plot_data)
plt.legend()
plt.savefig('picture/pre_true_pred.png')
plt.show()

plot_data = pd.DataFrame(data={
    'feat_name': feat_importance,
    'Importance': importance
})
plot_data = plot_data.set_index(keys='feat_name', drop=True)
plot_data.plot(kind='bar')
plt.savefig('picture/pre-importance.png')
plt.show()


def sesuper():
    return r2, clf.history['loss']
