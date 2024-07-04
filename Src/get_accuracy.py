from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, log_loss
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def GBM(X_train,y_train):
    gbm_model = GradientBoostingClassifier(random_state=42)
    gbm_model.fit(X_train, y_train)
    return gbm_model

def LGBM(X_train,y_train):
    lgbm_model = LGBMClassifier(random_state=42, verbose=-1)
    lgbm_model.fit(X_train, y_train)
    return lgbm_model

def XGB(X_train,y_train):
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def CatBoost(X_train,y_train):
    catboost_model = CatBoostClassifier(verbose=0, random_state=42)
    catboost_model.fit(X_train, y_train)
    return catboost_model

def mode_pred(model, X):
    pred = model.predict(X)
    return pred

def plot_confusion_matrix(predictions, actuals, classes=['Negative', 'Positive']):
    cm = confusion_matrix(actuals, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=False,xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def evaluate_classification_model(predictions, actuals):
    acc = accuracy_score(actuals, predictions)
    prec = precision_score(actuals, predictions)
    rec = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)
    logloss = log_loss(actuals, predictions)
    
    # Print the results
    print("Evaluation Metrics:")
    print("Test Accuracy:", acc)
    print("="*60)
    print("Precision:", prec)
    print("="*60)
    print("Recall:", rec)
    print("="*60)
    print("F1 Score:", f1)
    print("="*60)
    print("Log Loss:", logloss)
    print("="*60)

    # Plot the confusion matrix
    print("Confusion Matrix:\n")
    plot_confusion_matrix(predictions, actuals)

def get_results(X_train, y_train, X_test, y_test):
    GBM_model = GBM(X_train, y_train)
    LGBM_model = LGBM(X_train, y_train)
    XGB_model = XGB(X_train, y_train)
    CatBoost_model = CatBoost(X_train, y_train)

    print("###################### GBM Model ######################")
    pred_train = mode_pred(GBM_model, X_train)
    acc = accuracy_score(y_train, pred_train)
    print(f"Train Accuracy: {acc}")
    predictions = mode_pred(GBM_model, X_test)
    evaluate_classification_model(predictions, y_test)

    print("###################### LGBM Model ######################")
    pred_train = mode_pred(LGBM_model, X_train)
    acc = accuracy_score(y_train, pred_train)
    print(f"Train Accuracy: {acc}")
    predictions = mode_pred(LGBM_model, X_test)
    evaluate_classification_model(predictions, y_test)

    print("###################### XGB Model ######################")
    pred_train = mode_pred(XGB_model, X_train)
    acc = accuracy_score(y_train, pred_train)
    print(f"Train Accuracy: {acc}")
    predictions = mode_pred(XGB_model, X_test)
    evaluate_classification_model(predictions, y_test)

    print("###################### CatBoost Model ######################")
    pred_train = mode_pred(CatBoost_model, X_train)
    acc = accuracy_score(y_train, pred_train)
    print(f"Train Accuracy: {acc}")
    predictions = mode_pred(CatBoost_model, X_test)
    evaluate_classification_model(predictions, y_test)