
import numpy as np
import pandas as pd
import phik
import time

from modules import dataprep as dp
from modules import featureselection as fs
from modules import models as mods
from modules import metrics as mt


## Run a few models for feature selection and  prediction (should try pca for feature selection also)
models = ['RandomForest', 'Adaboost', 'GradientBoostingClassifier']
FeatureSelection = ['SelectFromModel', 'SequentialFeatureSelector']
oversampling = [True, False]
for model in models:
    for featureSel_method in FeatureSelection:
        for IsOversampling in oversampling:
            start = time.time()
            X_train, X_test, y_train, y_test = dp.get_data('caravan-insurance-challenge.csv', IsOversampling)
            print (y_train['CARAVAN'].value_counts())
            
            
            rf_features = fs.rf_feature_selection(X_train, y_train, False,featureSel_method, model)
            rfmod, rfpreds = mods.rf(X_train, X_test, y_train, y_test, rf_features, model)
            print ('model: ', model, ' using ',featureSel_method, 'as feature selection method')
            print ('over sampling is: ',IsOversampling)
            mt.metrics(rfmod, rfpreds, y_test, False)
            
            print ('process time: ', (time.time()-start)/60, ' minutes')
            print ('.........................................Next result...............................................\n')


# Logistic Regerssion Model
# Prep Data
X_train, X_test, y_train, y_test = dp.get_data('caravan-insurance-challenge.csv', False)
lr_features = fs.lr_feature_selection(X_train, y_train, verbose=False)
lmmod, lmpreds = mods.lm(X_train, X_test, y_train, y_test, lr_features)
mt.metrics(lmmod, lmpreds, y_test, "Logistic Regression")


#Removing highly correlated features 
#model:  GradientBoostingClassifier  using  SequentialFeatureSelector as feature selection method
#over sampling is:  True
X_train, X_test, y_train, y_test = dp.get_data('caravan-insurance-challenge.csv', True)
correlated_features = set()
#correlation_matrix = X_train.corr()
correlation_matrix=X_train.phik_matrix()

for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            print(correlation_matrix.columns[i])
            print(correlation_matrix.columns[j])
            print("---")
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)

len(correlated_features)
print(correlated_features)
featureSel_method= 'SequentialFeatureSelector'
model = 'GradientBoostingClassifier'
IsOversampling=True
X_train.drop(labels=correlated_features, axis=1, inplace=True)
X_test.drop(labels=correlated_features, axis=1, inplace=True)
rf_features = fs.rf_feature_selection(X_train, y_train, True, featureSel_method, model)
rfmod, rfpreds = mods.rf(X_train, X_test, y_train, y_test, rf_features, model)
print ('model: ', model, ' using ',featureSel_method, 'as feature selection method')
print ('over sampling is: ',IsOversampling)
mt.metrics(rfmod, rfpreds, y_test, True)
