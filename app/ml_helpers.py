from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

DEFAULT_PKL_LOCATION = 'emotion_detector.pkl'

def train_ert(x, y):
    ert_classifier = ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_split=2, random_state=0)
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
    evaluate_models(get_classifiers(), x, y)

    return data

def get_classifiers():
    ada_boost_100 = AdaBoostClassifier(n_estimators=100)
    ada_boost_100_point5 = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
    ada_boost_100_point1 = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
    ada_boost_100_point01 = AdaBoostClassifier(n_estimators=100, learning_rate=0.01)
    ada_boost_200 = AdaBoostClassifier(n_estimators=200)
    ert_100 = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0, n_jobs=-1)
    ert_200 = ExtraTreesClassifier(n_estimators=200, max_depth=None, min_samples_split=2, random_state=0, n_jobs=-1)
    xgboost_100 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=None, random_state=0)
    xgboost_100_point5 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=None, random_state=0)
    xgboost_100_point1 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=None, random_state=0)
    xgboost_100_point01 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=None, random_state=0)
    centroid = NearestCentroid()
    neighbors_10 = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
    neighbors_20 = KNeighborsClassifier(n_neighbors=20, n_jobs=-1)
    logistic = LogisticRegression(solver='lbfgs', n_jobs=-1)
    perceptron = Perceptron(n_jobs=-1)
    perceptron_l2 = Perceptron(penalty='l2', n_jobs=-1)
    perceptron_elastic = Perceptron(penalty='elasticnet', n_jobs=-1)
    logistic_sgd = SGDClassifier(loss='log', penalty='elasticnet')
    perceptron_sgd = SGDClassifier(loss='perceptron', penalty='elasticnet')
    neural_net_hidden_10_10 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 10))
    neural_net_hidden_100 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,))
    neural_net_hidden_50_50 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50,50))

    classifiers = [ \
        (ada_boost_100, "AdaBoost 100"), \
        (ada_boost_100_point5, "AdaBoost 100 0.5"), \
        (ada_boost_100_point1, "AdaBoost 100 0.1"), \
        (ada_boost_100_point01, "AdaBoost 100 0.01"), \
        (ada_boost_200, "AdaBoost 200"), \
        (ert_100, "ERT 100"), \
        (ert_200, "ERT 200"), \
        (xgboost_100, "XGBoost 100"), \
        (xgboost_100_point5, "XGBoost 100 0.5"), \
        (xgboost_100_point1, "XGBoost 100 0.1"), \
        (xgboost_100_point01, "XGBoost 100 0.01"), \
        (centroid, "Nearest Centroid"), \
        (neighbors_10, "Nearest Neighbors 10"), \
        (neighbors_20, "Nearest Neighbors 20"), \
        (logistic, "Logistic Regression"), \
        (perceptron, "Perceptron"), \
        (perceptron_l2, "Perceptron L2"), \
        (perceptron_elastic, "Perceptron ElasticNet"), \
        (logistic_sgd, "Logistic w/ Gradient Descent"), \
        (perceptron_sgd, "Perceptron w/ Gradient Descent"), \
        (neural_net_hidden_10_10, "Neural Network 10 10"), \
        (neural_net_hidden_100, "Neural Network 100"), \
        (neural_net_hidden_50_50, "Neural Network 50 50") \
    ]

    return classifiers

def evaluate_models(classifiers, x, y):
    for classifier in classifiers:
        evaluate_model(classifier[0], classifier[1], x, y)

def evaluate_model(model, name, x, y):
    scores = cross_val_score(model, x, y)
    mean_accuracy = scores.mean()
    print("{name} mean accuracy is {accuracy}.".format(name=name, accuracy=mean_accuracy))

def predict_with_model(model):
    def predict_result(data):
        return model.predict(data)
    return predict_result