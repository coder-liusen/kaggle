# kaggle
kaggle competition code

This repository is used for storing python projects for kaggle competition, so far it only has a titanic competition in it.  
  
1.titanic:  
use pandas to preprocess the data,mostly discard the columns of the data in which the missing value exits,and transform str type to int;  
use GradientBoostingClassifier (GBDT model) to fit the data;  
and use GridSearchCV to search the best hyper parameter of the model.

tips of titanic:  
(1) simply discard the coulmns which missing some of it's values maybe an effective processing, because filling NaN with any strategy can lead to nonreal info;  
(2) hyper parameter tunning of model is so hard, sometimes you will get lower score on kaggle web even though you get a higher score on local dataset. it's so hard to estimate the exact correlation between hyper parameter tunning and generalization ability of the model.
