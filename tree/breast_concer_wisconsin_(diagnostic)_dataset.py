import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz

data = pd.read_csv('breast_concer_wisconsin_(diagnostic)_dataset.csv')
pd.set_option('display.max_column', 99)
data.columns
data['Unnamed: 32']
data.drop('Unnamed: 32', axis = 1, inplace = True)
data['malignant'] = np.where(data['diagnosis'] == 'M', 1, 0)
data.drop('diagnosis', axis = 1, inplace = True)
len(data[data['malignant'] == 1])/ len(data)
len(data[data['malignant'] == 0])/ len(data)
data.drop(data[data['malignant'] == 0].sample(frac = 1).index[:146], inplace = True)
data = data.sample(frac = 1)
data_train = data[:400]
data_test = data[400:]
data_train_M = data_train['malignant'] 
data_test_M = data_test['malignant'] 
data_train_a = data_train.drop('malignant', axis = 1)
data_test_a = data_test.drop('malignant', axis = 1)
model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth= 5)
model = model.fit(data_train_a, data_train_M)
export_graphviz(model, out_file = "model.dot")
import graphviz
with open("model.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
model.score(data_test_a, data_test_M)
data_a = data.drop('malignant', axis = 1)
data_M = data['malignant']
scoresall = cross_val_score(model, data_a, data_M, cv = 7)
print('Total Accuracy : %.2f (+/- %.2f)' % (scoresall.mean(), scoresall.std() * 2))
data_acc = np.empty((19,3), float)
i = 0
for max_depth in range(2, 21) :
    model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth= max_depth)
    score = cross_val_score(model, data_a, data_M, cv = max_depth)
    data_acc[i, 0] = max_depth
    data_acc[i, 1] = score.mean()
    data_acc[i, 2] = score.std() * 2
    i += 1
data_acc
plt.errorbar(data_acc[:, 0], data_acc[:, 1], yerr = data_acc[:, 2])

graphviz.Source(dot_graph)