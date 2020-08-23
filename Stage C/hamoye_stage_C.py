# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 16:44:46 2020

@author: ADELEKE OLADAPO
"""


import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold , StratifiedKFold, LeaveOneOut
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, recall_score,accuracy_score,precision_score,confusion_matrix



df = pd.read_csv('C:/Users/ADELEKE OLADAPO/atom projects/Hamoye Practice/Hamoye Project/Hamoye_Internship/Stage C/NFA 2019 public_data.csv', low_memory =False)


#### Lets also check the missing data
df.dropna(inplace=True)

# treating 1A attributes as 2A
df.QScore.replace('1A','2A', inplace=True)
df.QScore.value_counts()


df_2A = df[df.QScore =='2A']
df_3A = df[df['QScore']=='3A'].sample(350)

#merging the two
data = df_2A.append(df_3A)
data.reset_index(drop=True, inplace=True)
data.index=data.index +1


# shuffling the data
data = shuffle(data)
data.reset_index(inplace=True)
data.index = data.index+1

# splitting the data
x = data.drop(['country_code','QScore', 'country', 'year'], 1)
y = data.QScore

#to encode a needed column
encoder = LabelEncoder()

l = []
for i in x.forest_land:
    if type(i) is str:
        l.append(float(i))
    else:
        l.append(0.0)
x.forest_land= l

X_train, X_test, Y_train, Y_test= train_test_split(x,y,
                                                    random_state=0,
                                                  test_size = 0.3)
X_train.record = encoder.fit_transform(X_train.record)
X_test.record = encoder.transform(X_test.record)

#to check imbalance in the target data
smote = SMOTE(random_state=1)
x_balanced, y_balanced = smote.fit_resample(X_train,Y_train)


print(y_balanced.value_counts())
# Scaling the data, also called normalising data
# encoding the record columns cos we need it
log_reg= LogisticRegression()
maxscaler = MinMaxScaler()

norm_df = pd.DataFrame(maxscaler.fit_transform(x_balanced.drop('record',1)),
                       columns=x_balanced.drop('record',1).columns)
norm_df['record'] = x_balanced['record']

X_test.reset_index(drop=True,inplace=True)
test_df = pd.DataFrame(maxscaler.transform(X_test.drop('record',1)),
                        columns = X_test.drop('record',1).columns)
test_df['record'] = X_test.record
scores = cross_val_score(log_reg,
                            norm_df,
                            y_balanced,
                             cv=5,
                             scoring='f1_macro')
print('scores:\n',scores)

# LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(norm_df, y_balanced)


# KFold

kf = KFold(n_splits=5)
f_scores = []
for train_index,test_index in kf.split(norm_df):
    xtrain,xtest = norm_df.iloc[train_index],norm_df.iloc[test_index]
    ytrain,ytest = y_balanced.iloc[train_index],y_balanced.iloc[test_index]
    model = LogisticRegression().fit(xtrain,ytrain)
    f_scores.append(f1_score(ytest, model.predict(xtest), pos_label='2A')*100)

print('f_scores: \n',f_scores)
print(model.predict(xtest).shape)


# StratifiedKFold
skf = StratifiedKFold(n_splits= 5 , shuffle= True , random_state= 1 )
f1_scores = []
#run for every split
for train_index, test_index in skf.split(norm_df, y_balanced):
    x_train, x_test = np.array(norm_df)[train_index],np.array(norm_df)[test_index]
    y_train, y_test = y_balanced[train_index], y_balanced[test_index]
    modell = LogisticRegression().fit(x_train, y_train)
 #save result to list
    f1_scores.append(f1_score(y_true=y_test, y_pred=modell.predict(x_test), pos_label= '2A' ))
print('f1_scores: \n', f1_scores)


# LeaveOneOut
Loo = LeaveOneOut()
score = cross_val_score(LogisticRegression(), norm_df,y_balanced,
                        cv=Loo,
                        scoring= 'f1_macro')

average_score = score.mean()*100
print('LeaveOneOut mean score: ', average_score)


# confusion_matrix
prediction = log_reg.predict(test_df)
matrix = confusion_matrix(Y_test, prediction, labels=['2A','3A'])
print(matrix)

# Accuracy
acc  = accuracy_score(Y_test,prediction)
print('Accuracy of the prediction:', round(acc*100,2))

# precision
precision = precision_score(Y_test, prediction, pos_label='2A')
print('precision: \n', round(precision*100,2))


# Recall
recall = recall_score(Y_test, prediction, pos_label= '2A')
print("Recall score: ", round(recall*100, 2))


# F1_score
f1 = f1_score(Y_test, prediction, pos_label='2A')
print('f1_score: ', round(f1*100,2))
