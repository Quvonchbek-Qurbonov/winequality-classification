from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from app.services.data import Data
import timeit
from app.schemas.schema import Metrics


def ordinary_training(trash_data: bool = False):  #just training without hyperparameter tuning
    if not trash_data:
        X, y = Data.get_data()
    else:
        X, y = Data.get_broken_data()
    X, y = Data.preprocess_data(X, y)

    model = SVC(
        kernel='linear'
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=55
    )

    start_time = timeit.default_timer()
    model.fit(X_train, y_train)
    training_time = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    y_pred = model.predict(X_test)
    test_time = timeit.default_timer() - start_time

    metrics = Metrics(
        trainingTime=training_time,
        testingTime=test_time,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, average='macro'),
        recall=recall_score(y_test, y_pred, average='macro'),
        f1=f1_score(y_test, y_pred, average='macro')
    )

    return metrics

def hyperparameter_tuning(trash_data: bool = False):
    if not trash_data:
        X, y = Data.get_data()
    else:
        X, y = Data.get_broken_data()

    X, y = Data.preprocess_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=55
    )

    model = SVC()

    # param_grid = {
    #     'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    #     'C': [0.0001, 0.001,0.01, 0.1, 1, 2],
    #     'gamma': [0.0001, 0.001,0.01, 0.1, 1, 2],
    #     'degree': [2,3],
    #     'class_weight': [None, 'balanced'],
    # }

    param_grid = [
        # Grid 1: RBF and Sigmoid (tuned with C and gamma)
        {
            'kernel': ['rbf', 'sigmoid'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'class_weight': [None, 'balanced']
        },
        # Grid 2: Linear (only requires C)
        {
            'kernel': ['linear'],
            'C': [0.1, 1, 10],
            'class_weight': [None, 'balanced']
        },
        # Grid 3: Polynomial (requires degree and gamma)
        {
            'kernel': ['poly'],
            'C': [0.1, 1, 10],
            'degree': [2, 3],
            'gamma': ['scale', 'auto'],
            'class_weight': [None, 'balanced']
        }
    ]

    model_hyper = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        refit=True
    )

    start_time = timeit.default_timer()
    model_hyper.fit(X_train, y_train)
    training_time = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    y_pred = model_hyper.predict(X_test)
    test_time = timeit.default_timer() - start_time

    metrics = Metrics(
        trainingTime=training_time,
        testingTime=test_time,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, average='macro'),
        recall=recall_score(y_test, y_pred, average='macro'),
        f1=f1_score(y_test, y_pred, average='macro')
    )
    print(model_hyper.best_params_)
    return metrics
