from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def metrics(mod, preds, y_test, model_name, full=True):
    print(model_name)
    print(confusion_matrix(y_test, preds))
    
    if(full):
        print(classification_report(y_test, preds))
    else:
        print("Accuracy_score: " + str(round(accuracy_score(y_test, preds),2)))