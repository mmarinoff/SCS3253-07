import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
from utils import import_data, process_data, model_update_results, import_clusters
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


def cv_scores(ml_func, X_train, y_train, testname=None, ):
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    scores = []

    for train, test in skf.split(X_train, y_train):
        ml_func.fit(X_train.iloc[train, :], y_train.iloc[train])

        y_true = y_train.iloc[test]
        y_pred = ml_func.predict(X_train.iloc[test, :])
        acc = accuracy_score(y_true, y_pred)
        precision, recall, fbeta, support = precision_recall_fscore_support(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        scores.append([mse, acc, precision[1], recall[1], fbeta[1]])

    scores = np.array(scores).sum(axis=0)/5
    print(scores[1])
    model_update_results(scores, modelname=testname)


def logistic_regression(X_train, y_train, optimise=False):

    if optimise:
        # Best parameter(CV score = 0.914): {'logistic__C': 0.9142857142857143, 'logistic__solver': 'sag'}
        pipe = Pipeline(steps=[
            # ('stdscale', StandardScaler())
            ('logistic', LogisticRegression())
        ])

        param_grid = {
            'logistic__solver': ['liblinear'],
            'logistic__solver': ['sag', 'saga', 'newton-cg'],
            'logistic__C': np.linspace(0.7, 1.0, 15),
        }
        pipe.fit(X_train, y_train)  # apply scaling on training data

        # Performance Metrics
        search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=10)
        search.fit(X_train, y_train.ravel())
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)

    else:
        log = LogisticRegression(C=0.9, solver='sag', penalty='l2')  # best from cross validation
        cv_scores(log, X_train, y_train, testname='Baseline Logistic 2')


def random_forest(X_train, y_train, meta_train, optimise=False):
    # best results ccp_alpha=0, criterion='entropy', max_features='log2', min_samples_split=5}
    # Best parameter(CV score = 0.918):{'random_forest__ccp_alpha': 0, 'random_forest__criterion': 'entropy',
    # 'random_forest__max_features': 'log2', 'random_forest__min_samples_split': 5}

    feature_cluster = FeatureAgglomeration(n_clusters=20)  # reduce dimensions to 20
    X_train = pd.DataFrame(feature_cluster.fit_transform(X_train))  # reduce samples to

    X_train = pd.concat([X_train, meta_train], axis=1)

    if optimise:

        pipe = Pipeline(steps=[
            ('random_forest', RandomForestClassifier())
        ])

        param_grid = {
            'random_forest__criterion': ['gini', 'entropy'],
            'random_forest__min_samples_split': [5],
            'random_forest__ccp_alpha': [0, 0.1],
            'random_forest__max_features': ['auto', 'sqrt', 'log2']
        }
        pipe.fit(X_train, y_train)  # apply scaling on training data

        # Performance Metrics
        search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=10)
        search.fit(X_train, y_train.ravel())
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)

    else:
        # best from cross validation
        rfc = RandomForestClassifier(criterion='entropy', min_samples_split=5, ccp_alpha=0, max_features='log2')
        cv_scores(rfc, X_train, y_train, testname='metafeatures random forest')


def grad_boost(X_train, y_train, meta_train, optimise=False):
    # Feature Reduction
    # data_minmax = MinMaxScaler().fit_transform(data_x)
    # kbest = SelectKBest(f_classif, k='all').fit(data_minmax, data_y)
    #
    # kbest_scores = pd.DataFrame(kbest.scores_)
    # kbest_columns = kbest_scores >= 15
    # data_x = data_x.iloc[:, kbest_columns.values.flatten()]

    feature_cluster = FeatureAgglomeration(n_clusters=20)  # reduce dimensions to 20
    X_train = pd.DataFrame(feature_cluster.fit_transform(X_train))  # reduce samples to

    X_train = pd.concat([X_train, meta_train], axis=1)

    if optimise:
        # {criterion='friedman_mse', learning_rate=0.1, min_samples_leaf=10, min_samples_split=5,
        # n_estimators=200, subsample=0.5}
        # accuracy 0.9076491228070176
        pipe = Pipeline(steps=[
            ('gradboost', GradientBoostingClassifier())
        ])

        param_grid = {
            'gradboost__n_estimators': [200],
            'gradboost__subsample': [0.5, 0.7, 1],
            'gradboost__learning_rate': [0.1, 0.3],
            'gradboost__criterion': ['friedman_mse', 'mse'],
            'gradboost__min_samples_split': [5 ],
            'gradboost__min_samples_leaf': [2, 5]
        }
        pipe.fit(X_train, y_train)  # apply scaling on training data

        # Performance Metrics
        search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=10)
        search.fit(X_train, y_train.ravel())
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)

    else:
        # best from cross validation
        grad = GradientBoostingClassifier(criterion='friedman_mse', learning_rate=0.1, min_samples_leaf=10,
                                          min_samples_split=5, n_estimators=200, subsample=0.5)
        cv_scores(grad, X_train, y_train, testname='gradboost metafeatures')


def adaboost(X_train, y_train, meta_train, optimise=False):

    feature_cluster = FeatureAgglomeration(n_clusters=20)  # reduce dimensions to 20
    X_train = pd.DataFrame(feature_cluster.fit_transform(X_train))  # reduce samples to

    X_train = pd.concat([X_train, meta_train], axis=1)

    if optimise:
        # {'adaboost__algorithm': 'SAMME.R', 'adaboost__learning_rate': 1, 'adaboost__n_estimators': 100}
        # 0.9112
        pipe = Pipeline(steps=[
            ('adaboost', AdaBoostClassifier())
        ])

        param_grid = {
            'adaboost__n_estimators': [50, 100],
            'adaboost__algorithm': ['SAMME', 'SAMME.R'],
            'adaboost__learning_rate': [0.1, 0.5, 0.7, 1, 1.2, 1.5],
        }
        pipe.fit(X_train, y_train)  # apply scaling on training data

        # Performance Metrics
        search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=10)
        search.fit(X_train, y_train.ravel())
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)

    else:
        # best from cross validation
        grad = AdaBoostClassifier(algorithm='SAMME.R', learning_rate=1, n_estimators=100)
        cv_scores(grad, X_train, y_train, testname='adaboost metafeatures')


def naive_bayes_gaussian(X_train, y_train, meta_train, optimise=False):
    # best results ccp_alpha=0, criterion='entropy', max_features='log2', min_samples_split=5}
    # Best parameter(CV score = 0.918):{'random_forest__ccp_alpha': 0, 'random_forest__criterion': 'entropy',
    # 'random_forest__max_features': 'log2', 'random_forest__min_samples_split': 5}

    kbest = SelectKBest(f_classif, k='all').fit(X_train, y_train)

    kbest_scores = pd.DataFrame(kbest.scores_)
    kbest_columns = kbest_scores >= 150
    X_train = X_train.iloc[:, kbest_columns.values.flatten()]
    #
    # feature_cluster = FeatureAgglomeration(n_clusters=20)  # reduce dimensions to 20
    # X_train = pd.DataFrame(feature_cluster.fit_transform(X_train))  # reduce samples to
    # X_train.reset_index(inplace=True, drop=True)
    # meta_train.reset_index(inplace=True, drop=True)
    #X_train = pd.concat([X_train, meta_train], axis=1)


    if optimise:

        pipe = Pipeline(steps=[
            ('nb_gaus', GaussianNB())
        ])

        param_grid = {
            'nb_gaus__var_smoothing': ['0.00000001', '0.00000001'],
        }
        pipe.fit(X_train, y_train)  # apply scaling on training data

        # Performance Metrics
        search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=10)
        search.fit(X_train, y_train.ravel())
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)

    else:
        # best from cross validation
        gnb = GaussianNB()
        cv_scores(gnb, X_train, y_train, testname='gaussian naive no scale, kbest')


def naive_bayes_bernoulli(X_train, y_train, meta_train, optimise=False):

    kbest = SelectKBest(f_classif, k='all').fit(X_train, y_train)

    kbest_scores = pd.DataFrame(kbest.scores_)
    kbest_columns = kbest_scores >= 15
    X_train = X_train.iloc[:, kbest_columns.values.flatten()]
    #
    # feature_cluster = FeatureAgglomeration(n_clusters=20)  # reduce dimensions to 20
    # X_train = pd.DataFrame(feature_cluster.fit_transform(X_train))  # reduce samples to

    X_train = pd.concat([X_train, meta_train], axis=1)

    if optimise:

        pipe = Pipeline(steps=[
            ('bnb', BernoulliNB())
        ])

        param_grid = {
            'bnb__alpha': ['1', '0.5', '0'],
        }
        pipe.fit(X_train, y_train)  # apply scaling on training data

        # Performance Metrics
        search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=10)
        search.fit(X_train, y_train.ravel())
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)

    else:
        # best from cross validation
        cnb = BernoulliNB()
        cv_scores(cnb, X_train, y_train, testname='Bernouli clusters')


def svm(X_train, y_train, meta_train, optimise=False):

    X_train = pd.concat([X_train, meta_train], axis=1)

    if optimise:

        pipe = Pipeline(steps=[
            # ('stdscale', StandardScaler())
            ('SVC', SVC())
        ])

        param_grid = {
            'SVC__C': [0.1, 0.5, 1, 1.5, 2],
            'SVC__gamma': ['scale', 'auto'],
            'SVC__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        }

        # Performance Metrics
        search = GridSearchCV(pipe, param_grid, n_jobs=-1, verbose=10)
        search.fit(X_train, y_train.ravel())
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)

    else:
        # best from cross validation
        svc = SVC()
        cv_scores(svc, X_train, y_train, testname='SVC')


def clusters_testing(X_train, y_train, meta_train):
    clusters_train, clusters_test = import_clusters()

    columns = list(range(0, 200))
    X_train.columns = columns

    for i in range(0, 2):
        X_c = X_train[clusters_train.iloc[:, 1] == i]
        y_c = y_train[clusters_train.iloc[:, 1] == i]
        m_c = meta_train[clusters_train.iloc[:, 1] == i]
        print(y_c.shape)

        naive_bayes_gaussian(X_c, y_c, m_c, optimise=False)

X_train, y_train, X_test = import_data()

# standard scale, create metafeatures
meta_train, meta_test = process_data(X_train, y_train, X_test)



# X_train = pd.concat([X_train, meta_train], axis=1)
# logistic_regression(X_train, y_train)
# random_forest(X_train, y_train, meta_train, optimise=False)
# grad_boost(X_train, y_train, meta_train, optimise=False)
naive_bayes_gaussian(X_train, y_train, meta_train, optimise=False)
# adaboost(X_train, y_train, meta_train, optimise=False)
# naive_bayes_bernoulli(X_train, y_train, meta_train, optimise=False)
# svm(X_train, y_train, meta_train, optimise=True)

# clusters_testing(X_train, y_train, meta_train)



