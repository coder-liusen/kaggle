

'''
kaggle Titanic competition
just looking
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']


## 加载数据
df = pd.read_csv('data\\train.csv')


## 提取有用字段
df = df.loc[:,('Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked')]


## 删除空值（Age字段有空值）
df = df.dropna(how = 'any')


## 将罹难者和幸存者人数统一

df_0 = df[df.Survived == 0]
df_1 = df[df.Survived == 1]
df_0 = df_0[:df_1.shape[0]]


## 查看各字段下，幸存者和罹难者的人数分布（Age和Fare用直方图，其余用条形图）

def plot_bar(lst_0,lst_1,label,str_opt):

    dic_0 = {key:lst_0.count(key) for key in label}
    dic_1 = {key:lst_1.count(key) for key in label}

    x = np.arange(len(dic_0))
    y_0 = dic_0.values()
    y_1 = dic_1.values()

    a = plt.bar(x-0.2 , y_0 , 0.4 ,  color='dodgerblue' , label="罹难者",align='center')
    b = plt.bar(x+0.2 , y_1 , 0.4 ,  color='orangered' , label="幸存者",align='center')

    for i in a + b:
        h = i.get_height()
        plt.text(i.get_x() + i.get_width() / 2, h, '%d' % int(h), ha='center', va='bottom')

    plt.tick_params(labelsize=10)
    plt.xticks(x,label)

    plt.grid(linestyle=':', axis='y')
    plt.title(str_opt+'分布图',fontsize=16)
    plt.xlabel(str_opt,fontsize=16)
    plt.ylabel('人数',fontsize=16)
    plt.legend()
    plt.show()

    
def plot_hist(s_0,s_1,bins,ran,str_opt):

    plt.hist(s_0,bins,alpha=0.5,label="罹难者")
    plt.hist(s_1,bins,alpha=0.5,label="幸存者")
    

    plt.grid(linestyle=':', axis='y')
    plt.title(str_opt+'分布',fontsize=16)
    plt.xlabel(str_opt,fontsize=16)
    plt.ylabel('人数',fontsize=16)
    plt.legend()
    plt.xlim(ran)

    plt.show()


opt = 'Embarked'

if(opt == 'Pclass'):

    lst_0 = df_0.Pclass.to_list()
    lst_1 = df_1.Pclass.to_list()
    label = [1,2,3]
    plot_bar(lst_0,lst_1,label,'Pclass')

if(opt == 'Sex'):

    lst_0 = df_0.Sex.to_list()
    lst_1 = df_1.Sex.to_list()
    label = df_1.Sex.unique()
    plot_bar(lst_0,lst_1,label,'Sex')

if(opt == 'SibSp'):

    lst_0 = df_0.SibSp.to_list()
    lst_1 = df_1.SibSp.to_list()
    label = [0,1,2,3,4,5]
    plot_bar(lst_0,lst_1,label,'SibSp')

if(opt == 'Parch'):

    lst_0 = df_1[df_1.Survived == 0].Parch.to_list()
    lst_1 = df_1[df_1.Survived == 1].Parch.to_list()
    label = [0,1,2,3,4,5,6]
    plot_bar(lst_0,lst_1,label,'Parch')

if(opt == 'Embarked'):

    lst_0 = df_0.Embarked.to_list()
    lst_1 = df_1.Embarked.to_list()
    label = ['S', 'C', 'Q']
    plot_bar(lst_0,lst_1,label,'港口')
    
if(opt == 'Fare'):

    bins = np.arange(0,520,10)
    s_0 = df_0.Fare
    s_1 = df_1.Fare
    ran = (0,520) 
    plot_hist(s_0,s_1,bins,ran,"票价")

if(opt == 'Age'):

    bins = np.arange(0,80,10)
    s_0 = df_0.Age
    s_1 = df_1.Age
    ran = (0,80)
    plot_hist(s_0,s_1,bins,ran,"年龄")



    

