
'''
kaggle Titanic competition
simple GBDT model with no parameter tuning
'''

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


## 训练模型
y = df_train['Survived'].values # 样本标签
X = df_train.loc[:,('Sex','Pclass','SibSp','Parch','Fare','Embarked')].values # 样本特征

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X,y)


## 预测
X_test = df_test.loc[:,('Sex','Pclass','SibSp','Parch','Fare','Embarked')]
y_pre = model.predict(X_test)


## 保存数据
df_test['Survived']=y_pre
df = df_test.loc[:,('PassengerId','Survived')]
df.to_csv('data\\submission_0.csv',index = None)

