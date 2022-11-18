import pandas as pd
import os

os.chdir(r"D:\PyCharm 2021.2.2\data\Python学习笔记\机器学习算法入门\n.自然语言处理\词项共现网络")

data = pd.read_csv('./main/original_data.csv', encoding='ANSI')[0:]
data['描述'] = data['描述'].apply(lambda x: x.replace('\n', '').replace('\r', '').replace(' ', ''))
data_new = data['描述']
data_new = data_new.dropna(axis=0)
data_new = data_new.drop_duplicates()
print(data_new)
# data_new.to_csv('./data/data2.txt', encoding='utf-8', index=False, header=False)

data = pd.read_csv('./main/original_data.csv', encoding='ANSI')[0:]
data['房客评价'] = data['房客评价'].apply(lambda x: str(x).replace('\n', '').replace('\r', '').replace(' ', ''))
data_new = data['房客评价']
data_new = data_new.dropna(axis=0)
data_new = data_new.drop_duplicates()
print(data_new)
# data_new.to_csv('./data/data3.txt', encoding='utf-8', index=False, header=False)
