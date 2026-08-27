from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

    model = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        class_weight='balanced'
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

    model = DecisionTreeClassifier()

    param_grid = {
        'criterion': ['gini', 'entropy', 'logloss'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 2, 5, 10, 20],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [2, 5, 10, 20],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'min_impurity_decrease': [0.0, 0.00001, 0.0002, 0.003, 0.04, 0.1],
        'class_weight': [None, 'balanced']
    }

    model_hyper = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=40,
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
