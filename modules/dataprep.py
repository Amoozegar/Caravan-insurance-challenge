import pandas as pd
from imblearn.over_sampling import SMOTE

def get_data(file, oversampling):
    
    data = pd.read_csv(file)
    train= data[data['ORIGIN']=='train']
    test= data[data['ORIGIN']=='test']

    X_train = train.iloc[:,1:-1]
    X_test  = test.iloc[:,1:-1]
    y_train = train.iloc[:,-1:]
    y_test  = test.iloc[:,-1:]
    if oversampling== True:
        sm = SMOTE()
        X_train2, y_train2 = sm.fit_sample(X_train, y_train)
    if oversampling== False:
        X_train2 = X_train 
        y_train2 = y_train
    print('Training Set Shape after oversampling:   ', X_train2.shape, y_train2.shape)
    return X_train2, X_test, y_train2, y_test