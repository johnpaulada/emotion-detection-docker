from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

DEFAULT_PKL_LOCATION = 'emotion_detector.pkl'

def train_ert(x, y):
    ert_classifier = ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_split=2, n_jobs=-1)
    ert_classifier.fit(x, y)

    return ert_classifier

def train_model(data):
    print("Training...")
    x, y = data

    return train_ert(x, y)

def save_model(model, location=DEFAULT_PKL_LOCATION):
    print("Saving model...")
    joblib.dump(model, location)

    return model

def load_model(location=DEFAULT_PKL_LOCATION):
    return joblib.load(location)

def experiment(data):
    print("Experimenting...")
    x, y = data
    results = evaluate_models(get_classifiers(), x, y)
    model, score = get_best(results)
    print("Best model is {model} with a score of {score}.".format(model=model, score=score))

    return data

def get_best(results):
    best = ('', 0)
    for result in results:
        if result[1] > best[1]:
            best = result
    return best

def get_classifiers():
    # adaboost_params = { \
    #     'n_estimators': (100, 200, 500), \
    #     'learning_rate': (1, 0.5, 0.1, 0.01) \
    # }
    # adaboost = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=adaboost_params, n_jobs=-1, refit=True)
    adaboost = AdaBoostClassifier(n_estimators=500, learning_rate=0.5)

    # ert_params = { \
    #     'n_estimators': (200, 300), \
    #     'criterion': ('gini', 'entropy'), \
    #     'max_depth': (None,), \
    #     'min_samples_split': (2, 3, 4), \
    #     'n_jobs': (-1,) \
    # }
    # ert = GridSearchCV(estimator=ExtraTreesClassifier(), param_grid=ert_params, n_jobs=-1)
    ert = ExtraTreesClassifier(n_estimators=200, min_samples_split=2, max_depth=None, n_jobs=-1)

    # xgboost_params = { \
    #     'n_estimators': (100, 200, 500), \
    #     'learning_rate': (1, 0.5, 0.1, 0.01), \
    #     'max_features': ('sqrt', 'log2'), \
    #     'presort': ('auto',), \
    #     'max_depth': (None,) \
    # }
    # xgboost = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid=xgboost_params, n_jobs=-1)
    xgboost = GradientBoostingClassifier(n_estimators=100, max_features='sqrt', presort='auto', learning_rate=0.5, max_depth=None)

    centroid = NearestCentroid()
    neighbors_5 = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    neighbors_10 = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
    logistic = LogisticRegression(solver='lbfgs', n_jobs=-1)
    perceptron_elastic = Perceptron(penalty='elasticnet', max_iter=1000, tol=1e-3, n_jobs=-1)

    # mlp_params = { \
    #     'solver': ('lbfgs',), \
    #     'activation': ('tanh', 'relu'), \
    #     'hidden_layer_sizes': ((100,100,100), (200,200,200)), \
    #     'alpha': (1e-3, 1e-5), \
    # }
    # neural_net = GridSearchCV(estimator=MLPClassifier(), param_grid=mlp_params, n_jobs=-1)
    neural_net = MLPClassifier(solver='lbfgs', activation='tanh', hidden_layer_sizes=(500, 500, 500, 500))

    classifiers = [ \
        (adaboost, "AdaBoost"), \
        (ert, "ERT"), \
        (xgboost, "XGBoost"), \
        (centroid, "Nearest Centroid"), \
        (neighbors_5, "Nearest Neighbors 5"), \
        (neighbors_10, "Nearest Neighbors 10"), \
        (logistic, "Logistic Regression"), \
        (perceptron_elastic, "Perceptron ElasticNet"), \
        (neural_net, "Neural Network") \
    ]

    return classifiers

def evaluate_models(classifiers, x, y):
    return [(classifier[1], evaluate_model(classifier[0], classifier[1], x, y)) for classifier in classifiers]

def evaluate_model(model, name, x, y):
    print("=========")
    print('Processing {name}...'.format(name=name))
    if hasattr(model, 'best_score_') and hasattr(model, 'best_params_'):
        model.fit(x, y)
        print("Params: ")
        print(model.best_params_)
        print('{name} best score is {score}.'.format(name=name, score=model.best_score_))
        return model.best_score_
    else:
        scores = cross_val_score(model, x, y)    
        mean_accuracy = scores.mean()
        print("Result:")
        print('{name} mean accuracy is {accuracy}.'.format(name=name, accuracy=mean_accuracy))
        return mean_accuracy

def predict_with_model(model):
    def predict_result(data):
        return model.predict(data)
    return predict_result