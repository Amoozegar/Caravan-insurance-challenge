from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


def rf(X_train, X_test, y_train, y_test, selected_feat, model):
    if model =='RandomForest':
        mod = RandomForestClassifier(n_estimators=105, max_depth=25, n_jobs=-1)
        mod.fit(X_train[selected_feat], y_train.values.ravel())

        preds = mod.predict(X_test[selected_feat])
        
    elif (model=='Adaboost'):
        mod = AdaBoostClassifier(n_estimators=150)
        mod.fit(X_train[selected_feat], y_train.values.ravel())

        preds = mod.predict(X_test[selected_feat])
    elif (model=='GradientBoostingClassifier'):
        mod = AdaBoostClassifier(n_estimators=150)
        mod.fit(X_train[selected_feat], y_train.values.ravel())

        preds = mod.predict(X_test[selected_feat])
        
    return mod, preds

def lm(X_train, X_test, y_train, y_test, selected_feat):

    mod = LogisticRegression(penalty = "l2", n_jobs=-1,max_iter = 10000)
    #mod = LogisticRegression(penalty = "l2", n_jobs=-1,max_iter = 10000,solver = 'liblinear')
    #mod = LogisticRegression(penalty = "elasticnet", n_jobs=-1,max_iter = 10000,solver = 'saga',l1_ratio=.5)
    mod.fit(X_train[selected_feat], y_train.values.ravel())

    preds = mod.predict(X_test[selected_feat])
    return mod, preds
