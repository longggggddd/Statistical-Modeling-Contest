# -*- coding:utf-8-*-
import numpy as np
import pandas as pd
import tqdm
from utils import kfold_stats_feature, mean_score

data = pd.read_csv('data/original_data.csv', encoding='ANSI')
pd.set_option('display.max_columns', None)

# 初步处理特征
data['描述'] = [len(str(i).strip().replace('\n', '')) for i in data['描述']]
data['价格'] = [float(i) for i in data['价格']]
data['房源类型'] = [str(i).strip().replace('\n', '').replace(' ', '') for i in data['房源类型']]
data['出租类型'] = [str(i).strip().replace('\n', '').replace(' ', '') for i in data['出租类型']]
data['可住人数'] = [int(i[0]) for i in data['可住人数']]
data['面积'] = [int(i.replace('平米', '')) for i in data['面积']]
data['省份'] = [str(i).strip().replace('\n', '').replace(' ', '') for i in data['省份']]
data['城区'] = [str(i).strip().replace('\n', '').replace(' ', '') for i in data['城区']]
data['房源评论数'] = [float(i) for i in data['房源评论数']]
data['房源综合得分'] = [float(i) for i in data['房源综合得分']]
data['房主好评率'] = [float(i.replace('%', '')) / 100 for i in data['房主好评率']]
data['房主回复率'] = [float(i.replace('%', '')) / 100 for i in data['房主回复率']]
data['房主接单率'] = [float(i.replace('%', '')) / 100 for i in data['房主接单率']]
data['房源好评率'] = [float(i.replace('%', '')) / 100 for i in data['房源好评率']]
data['近期预定量'] = [int(i.replace(' ', '').replace('晚', '')) for i in data['近期预定量']]
data['房客评价'] = data['房客评价'].fillna(0)

data_target_encode = kfold_stats_feature(data, ['房源类型', '出租类型', '省份', '城区'], '近期预定量', 4, 2022)

lst = []
for unit in tqdm.tqdm(data_target_encode['房客评价']):
    if unit != 0:
        l_1 = []
        sen_lst = str(unit).split('+')
        for i in sen_lst:
            score = mean_score(i)
            l_1.append(score)
        lst.append(np.mean(l_1))
    else:
        lst.append(np.nan)
data_target_encode['房客评价'] = lst
data_target_encode.to_excel('mid_data.xlsx')
