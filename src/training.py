from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

def train_models(folds,
                 model_names=['lr','dt','rf','xgb','cbc'],
):
    # Initialise Models
    lr = LogisticRegression(max_iter=200)
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    xgb = XGBClassifier()
    cbc = CatBoostClassifier()

    untrained_models = {
        'lr': lr,
        'dt': dt,
        'rf': rf,
        'xgb': xgb,
        'cbc': cbc,
    }

    # Initialise output lists
    models = {
        'lr': [],
        'dt': [],
        'rf': [],
        'xgb': [],
        'cbc': [],
    }

    for fold in folds:
        for name in model_names:
            untrained_models[name].fit(fold['X_train'], fold['y_train'])
            models[name].append(untrained_models[name])

    return models
    

def metrics(folds, models, model_names):
    #f1scores = {}
    f1score_max = 0
    for name in model_names:
        #f1scores[name] = []
        for k in range(len(folds)):
            if isinstance(models[name][k].predict(folds[k]['X_test'])[0], str):
                y_pred = (models[name][k].predict(folds[k]['X_test'])=='True')
            else:
                y_pred = models[name][k].predict(folds[k]['X_test'])
            y_true = folds[k]['y_test']
            acc = accuracy_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            f1score = f1_score(y_true, y_pred)
            #f1scores[name].append(f1score)
            if f1score > f1score_max:
                f1score_max = f1score
                best_model = {'name': name,
                              'fold': k,
                              'acc': acc,
                              'f1score': f1score,
                              'prec': prec,
                              'recall': recall,
                }
                
            print(f"Model '{name}' fold {k} Accuracy: {acc}, Precision: {prec}, Recall: {recall}, F1-Score: {f1score}.")

    print(f"The best model is '{best_model['name']}' with fold {best_model['fold']} Accuracy: {best_model['acc']}, Precision: {best_model['prec']}, Recall: {best_model['recall']}, F1-Score: {best_model['f1score']}.")
    return models[best_model['name']][best_model['fold']]