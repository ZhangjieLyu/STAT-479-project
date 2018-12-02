import numpy as np
import pandas as pd
data = pd.read_csv('mydata.csv')
## first introduce small sample
data1 = data.iloc[0:500,:]

del data1['Unnamed: 0']
X = data1
X = X.drop(['y'],axis = 1)
y = data1['y']

##  split data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0,stratify = y)

##--------------------------- Adaboost benchmark
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier()
clf.fit(X_train,y_train)
pre = clf.predict(X_test)

## accuracy
acc = np.mean(pre == y_test)
acc

### GridSearch
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

DTC = DecisionTreeClassifier(random_state = 1, max_features = 'log2',class_weight = 'balanced')

Ada = AdaBoostClassifier(base_estimator = DTC)


param_grid = [{'algorithm':['SAMME','SAMME.R'],'n_estimators':list(np.arange(40,60)),
               'base_estimator__criterion' : ['gini','entropy'],
               'base_estimator__splitter' : ['best','random']}]

gs = GridSearchCV(estimator = Ada,param_grid = param_grid,
                  refit = True)
gs.fit(X_train,y_train)
gs.best_estimator_
gs.best_score_
gs.best_params_
##-----------------------------XGboost benchmark
from xgboost import XGBClassifier
clf_xg = XGBClassifier()
clf_xg.fit(X_train,y_train)
pre_xg = clf_xg.predict(X_test)
acc_xg = np.mean(pre_xg == y_test)
acc


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(pre,y_test)
cm

















