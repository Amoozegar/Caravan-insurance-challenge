from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd

def rf_feature_selection(X_train, y_train, verbose, method, model):
    if (method=='SelectFromModel'and model=='RandomForest'):
        sel = SelectFromModel(RandomForestClassifier(n_estimators = 100, n_jobs=-1))
        sel.fit(X_train, y_train.values.ravel())
        selected_feat= X_train.columns[(sel.get_support())]
        if(verbose):
            print("Num features selected: " + str(len(selected_feat)))
            print(selected_feat)
            
    elif (method=='SelectFromModel'and (model=='Adaboost')):
        sel = SelectFromModel(AdaBoostClassifier(n_estimators=150))
        sel.fit(X_train, y_train.values.ravel())
        selected_feat= X_train.columns[(sel.get_support())]
        if(verbose):
            print("Num features selected: " + str(len(selected_feat)))
            print(selected_feat)
            
    elif (method=='SelectFromModel'and (model=='GradientBoostingClassifier')):
        sel = SelectFromModel(GradientBoostingClassifier(n_estimators=150))
        sel.fit(X_train, y_train.values.ravel())
        selected_feat= X_train.columns[(sel.get_support())]
        if(verbose):
            print("Num features selected: " + str(len(selected_feat)))
            print(selected_feat)
            
    elif (method=='SequentialFeatureSelector' and model=='RandomForest'):

        model = RandomForestClassifier(n_estimators = 100, n_jobs=-1)
        #Define RFE 
        rfe = RFE(model, n_features_to_select=50)
        rfe = rfe.fit(X_train, y_train.values.ravel())

        df_RFE_results = []
        for i in range(X_train.shape[1]):
            df_RFE_results.append(
                {      
                    'Feature_names': X_train.columns[i],
                    'Selected':  rfe.support_[i],
                    'RFE_ranking':  rfe.ranking_[i],
                }
            )
            
        df_RFE_results = pd.DataFrame(df_RFE_results)
        df_RFE_results.index.name='Columns'
        selected_feat=X_train.columns[df_RFE_results[df_RFE_results['Selected']==True].index]
    
    elif (method=='SequentialFeatureSelector' and model=='Adaboost'):

        model = AdaBoostClassifier(n_estimators=150)
        #Define RFE 
        rfe = RFE(model, n_features_to_select=50)
        rfe = rfe.fit(X_train, y_train.values.ravel())

        df_RFE_results = []
        for i in range(X_train.shape[1]):
            df_RFE_results.append(
                {      
                    'Feature_names': X_train.columns[i],
                    'Selected':  rfe.support_[i],
                    'RFE_ranking':  rfe.ranking_[i],
                }
            )
            
        df_RFE_results = pd.DataFrame(df_RFE_results)
        df_RFE_results.index.name='Columns'
        selected_feat=X_train.columns[df_RFE_results[df_RFE_results['Selected']==True].index]
    elif (method=='SequentialFeatureSelector' and (model=='GradientBoostingClassifier')):

        model = GradientBoostingClassifier(n_estimators=150)
        #Define RFE 
        rfe = RFE(model, n_features_to_select=50)
        rfe = rfe.fit(X_train, y_train.values.ravel())

        df_RFE_results = []
        for i in range(X_train.shape[1]):
            df_RFE_results.append(
                {      
                    'Feature_names': X_train.columns[i],
                    'Selected':  rfe.support_[i],
                    'RFE_ranking':  rfe.ranking_[i],
                }
            )
            
        df_RFE_results = pd.DataFrame(df_RFE_results)
        df_RFE_results.index.name='Columns'
        selected_feat=X_train.columns[df_RFE_results[df_RFE_results['Selected']==True].index]
    return selected_feat

def lr_feature_selection(X_train, y_train, verbose=True):
    sel = SelectFromModel(LogisticRegression(penalty = "l2", n_jobs=-1,max_iter = 10000))
    #sel = SelectFromModel(LogisticRegression(penalty = "l2", n_jobs=-1,max_iter = 10000,solver = 'liblinear'))
    #sel = SelectFromModel(LogisticRegression(penalty = "elasticnet", n_jobs=-1,max_iter = 10000,solver = 'saga',l1_ratio=.5))
    sel.fit(X_train, y_train.values.ravel())
    selected_feat= X_train.columns[(sel.get_support())]
    if(verbose):
        print("Num features selected: " + str(len(selected_feat)))
        print(selected_feat)
        #pd.series(sel.estimator_,feature_importances_,.ravel()).hist()
    return selected_feat