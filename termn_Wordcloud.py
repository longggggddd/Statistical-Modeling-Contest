# -*- coding:utf-8-*-
import numpy as np
import pandas as pd
from def_fun import term_f, paint

data = pd.read_csv('data/original_data.csv', encoding='ANSI')
data_describe = data['描述'].sample(frac=1.0)
data_reviews = data['房客评价'].dropna(axis=0).sample(frac=1.0)

data_describe = ' '.join(data_describe)
data_reviews = ' '.join(data_reviews)

term_f(30, data_describe)
paint(300, data_describe, name='describe')

term_f(30, data_reviews)
paint(300, data_reviews, name='reviews')
