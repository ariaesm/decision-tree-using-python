# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:38:32 2021

@author: Saeed
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn import tree
from sklearn.model_selection import cross_val_score
def condition(x):
    if x.split()[1] == 'hours':
        return int(x.split()[0]) / 24
    elif x.split()[1] == 'days':
        return int(x.split()[0])
    else :
        return 1 / 24
data = pd.read_csv('USA_cars_datasets.csv')
pd.set_option('display.max_columns', 999)
# =============================================================================
# Edit data
# =============================================================================
print(data.head())
print(data.columns)
print(data.info())
print(data.describe())
drop_columns = ['Unnamed: 0', 'vin', 'lot', 'country']
data.drop(drop_columns, axis = 1, inplace = True)
data.info()
data['insured']= np.where(data['title_status'] == 'clean vehicle', 0, 1)
data['count'] = np.where(True, 1, 1)
# =============================================================================
# color
# =============================================================================
data['color'].value_counts()
data['color'] = np.where((data['color'] == 'no_color' ) | 
                         (data['color'] == 'color:'), data['color'].mode(), data['color'])
# =============================================================================
# price
# =============================================================================
data['price'] = np.where(data['price'] < 1000, random.randrange(10000, 84000),
                         data['price'])

# =============================================================================
# condition
# =============================================================================
data['condition'] = data['condition'].apply(condition)
# data['condition'] = pd.to_numeric(data['condition'])
# =============================================================================
# analize
# =============================================================================
# fig = plt.figure()
# ax = fig.add_subplot(111)
plt.hist(data['price'], bins = 40)
plt.xlabel('price')
plt.ylabel('Count of cars')
data.boxplot(column ='price')
# +++++++++++++++++++++++++++++++++++++++++++++++++++
data_brand = data.groupby('brand').count()
data_brand.sort_values('count', ascending = False, inplace = True)
data_brand['count'].plot.bar()
plt.ylabel('count of cars')
# +++++++++++++++++++++++++++++++++++++++++++++++++++
data_state = data.groupby('state').count()
data_state.sort_values('count', ascending = False, inplace = True)
data_state['count'].plot.bar()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++
data_year = data.groupby('year').count()
data_year.sort_values('count', ascending = False, inplace = True)
data_year['count'].plot.bar()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++
brand_ = pd.crosstab(data.brand, data.insured.astype(bool))
brand_.plot(kind = 'bar', stacked = True, grid = False, color = ['red', 'blue'])
# ++++++++++++++++++++++++++++++++++++++++++++++++++++
year_ = pd.crosstab(data.year, data.insured.astype(bool))
year_.plot(kind = 'bar', stacked = True, grid = False, color = ['red', 'blue'])
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
data_color = data.groupby('color').count().sort_values('brand', ascending = False).head(12)[['count']]
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
fig = plt.figure()
g1 = fig.add_subplot(121)
g1.scatter(data.mileage, data.price, marker = '*')
g1.set_xlabel('mileage')
g2 = fig.add_subplot(122)
g2.hist(data.price)
g2.set_xlabel('price')
##############################################################
# =============================================================================
# pishbin
# =============================================================================
data = data.drop('title_status', axis = 1)
data = pd.get_dummies(data, columns = ['brand', 'model', 'year', 'color', 'state'])
data.columns
data = data.sample(frac = 1)
data_train = data[:2000]
data_test = data[2000:]
data_train_b = data_train['insured']
data_test_a = data_test.drop('insured', axis = 1)
data_train_a = data_train.drop('insured', axis = 1)
data_test_b = data_test['insured']
model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth= 5)
model = model.fit(data_train_a, data_train_b)
model.score(data_test_a, data_test_b)
data_a = data.drop('insured', axis = 1)
data_b = data['insured']
scoresall = cross_val_score(model, data_a, data_b, cv = 5)
print('Total Accuracy : %.2f (+/- %.2f)' % (scoresall.mean(), scoresall.std() * 2))