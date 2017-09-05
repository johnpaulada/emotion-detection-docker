from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score

DEFAULT_PKL_LOCATION = 'emotion_detector.pkl'

def train_ert(x, y):
    ert_classifier = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    ert_classifier.fit(x, y)

    return ert_classifier

def train_model(x, y):
    return train_ert(x, y)

def save_model(model, location=DEFAULT_PKL_LOCATION):
    joblib.dump(model, location)

def load_model(location=DEFAULT_PKL_LOCATION):
    return joblib.load(location)

def experiment(x, y):
    ada_boost_100 = AdaBoostClassifier(n_estimators=100)
    ert_10 = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    xgboost_100 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

    evaluate_model(ada_boost_100, "AdaBoost 100", x, y)
    evaluate_model(ert_10, "ERT 10", x, y)
    evaluate_model(xgboost_100, "XGBoost 100", x, y)

def evaluate_model(model, name, x, y):
    scores = cross_val_score(model, x, y)
    mean_accuracy = scores.mean()
    print("{name} mean accuracy is {accuracy}.".format(name=name, accuracy=mean_accuracy))