import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

df = pd.read_csv('Data_for_UCI_named.csv')
print(df.head())

# recall = 355/(355+1480)
# precision = 355/(355+45)
#
# f1 = 2*(recall*precision)/(recall+precision)

#to check for missing data
print(df.isnull().sum())
#zero missing data

# check the type of datatypes
print(df.info())

# stabf is an object type so we need to encode it to use standardscaler on it
# stabf_val = df.stabf.value_counts()
# l = []
# for i in df.stabf:
#     if i =='stable':
#         l.append(1)
#     if i == 'unstable':
#         l.append(0)
# df.stabf = l
# new_stab_val = df.stabf.value_counts()
# print(stabf_val, new_stab_val)
encoder = LabelEncoder()
df.stabf = encoder.fit_transform(df.stabf)
# to split the data now
features = df.drop(['stab', 'stabf'], 1)
target = df.stabf

X_train, X_test, y_train, y_test = train_test_split(features,target,
                                                    test_size=0.2,
                                                    random_state=1)

# using standardscaler
scaler = StandardScaler()
# training the scaler
x_train = scaler.fit_transform(X_train)
# scaling the test set`
x_test = scaler.transform(X_test)


# function to train a model

def Train_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_score = model.score(X_train,y_train)
    test_score = model.score(X_test,y_test)
    print(f'Train score : {train_score}, Test score: {test_score}\n')

    return(model)
# lightgbm
light = LGBMClassifier(random_state = 1)
print('LGBMClassifier')
light_ = Train_model(light, x_train, x_test, y_train, y_test)

# RandomForestClassifier
random_forest = RandomForestClassifier(random_state = 1)
print('RandomForestClassifier')
rforest = Train_model(random_forest, x_train, x_test, y_train, y_test)

# xgboost classifier
xgboost = XGBClassifier(random_state = 1)
print('xgboost')
xgb = Train_model(xgboost, x_train, x_test, y_train, y_test)

import warnings
warnings.filterwarnings('ignore')


extra_trees = ExtraTreesClassifier(random_state = 1)

params = {
    'n_estimators' : [50, 100, 300, 500, 1000],
    'min_samples_split' : [2, 3, 5, 7, 9],
    'min_samples_leaf' :[1, 2, 4, 6, 8],
    'max_features' : ['auto', 'sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(estimator=extra_trees,
                                    param_distributions=params,
                                    cv=5,n_iter=10,
                                    scoring='accuracy',
                                    n_jobs=-1,verbose=1,
                                    random_state=1)

random_search.fit(x_train,y_train)
# best parameters
print('best parameters\n',random_search.best_params_)

# score
print('best estimator\n',random_search.best_estimator_.score(x_test,y_test))

# retraining the extratrees
extra = ExtraTreesClassifier(random_state=1)
extra.fit(x_train,y_train)
print('extratree scores 2',extra.score(x_test,y_test))

# important features
data = pd.Series`(extra.feature_importances_, index = features.columns)
print(data.sort_values())
