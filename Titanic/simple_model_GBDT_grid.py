
import numpy as np
import pandas as pd

## 加载数据

df_train = pd.read_csv('data\\train.csv')
df_test = pd.read_csv('data\\test.csv')

## 数据预处理

df_train = df_train.loc[:,('Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked')]
df_train['Sex'] = df_train.Sex.replace({'male':1,'female':2})
df_train['Embarked'] = df_train.Embarked.replace({'S':1,'Q':2,'C':3})
df_train = df_train.fillna(value = 1)

df_train['SibSp'] = df_train.SibSp + 1
df_train['Parch'] = df_train.Parch + 1

df_test = df_test.loc[:,('PassengerId','Survived','Pclass','Sex','SibSp','Parch','Fare','Embarked')]
df_test['Sex'] = df_test.Sex.replace({'male':1,'female':2})
df_test['Embarked'] = df_test.Embarked.replace({'S':1,'Q':2,'C':3})
df_test = df_test.fillna(value = 1)

df_test['SibSp'] = df_test.SibSp + 1
df_test['Parch'] = df_test.Parch + 1


## 模型调参（网格调参）

y = df_train['Survived'].values # 样本标签
X = df_train.loc[:,('Sex','Pclass','SibSp','Parch','Fare','Embarked')].values # 样本特征

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

n_estimators_range = range(110,130,2) #126
max_depth_range = range(1,12) # 1
learning_rate_range = np.arange(0.1,1,0.1) # 0.2
min_samples_split_range = range(1,5) # 4


param_dic = {
             'n_estimators':n_estimators_range
##             'max_depth':max_depth_range
##             'learning_rate':learning_rate_range
##             'min_samples_split':min_samples_split_range\
             }

model = GradientBoostingClassifier()
grid = GridSearchCV(model,param_dic,cv = 3)
grid.fit(X,y)

print(grid.best_score_,grid.best_params_)





