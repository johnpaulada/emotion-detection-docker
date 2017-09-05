from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
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
        (xgboost_100_point01, "XGBoost 100 0.01") \
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