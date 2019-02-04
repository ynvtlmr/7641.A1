"""
Contains all the necessary functions to evaluate trained models and generate
validation, learning, iteration and timing curves.

"""
import pandas as pd
import numpy as np

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import timeit

from helpers import load_pickled_model, get_abspath
from model_train import split_data, balanced_accuracy
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve, StratifiedShuffleSplit, cross_val_score, validation_curve

# for decision boundary plotting
from mlxtend.plotting import plot_decision_regions

# for decision tree plotting
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus


def basic_results(grid, X_test, y_test, data_name, clf_name):
    """Gets best fit against test data for best estimator from a particular grid object.
    Note: test score function is the same scoring function used for training.

    Args:
        grid (GridSearchCV object): Trained grid search object.
        X_test (numpy.Array): Test features.
        y_test (numpy.Array): Test labels.
        data_name (str): Name of data set being tested.
        clf_name (str): Name of algorithm type.

    """
    # get best score, test score, scoring function and best parameters
    bs = grid.best_score_
    ts = grid.score(X_test, y_test)
    sf = grid.scorer_
    bp = grid.best_params_

    # write results to a combined results file
    parentdir = 'results'
    resfile = get_abspath('combined_results.txt', parentdir)

    # append grid score data to combined results csv
    with open(resfile, 'a') as f:
        f.write('{}|{}|{}|{}|{}|{}\n'.format(clf_name, data_name, bs, ts, sf, bp))


def create_learning_curve(
        estimator,
        scorer, X_train, y_train, data_name,
        clf_name, cv=5
):
    """Generates a learning curve for the specified estimator, saves tabular
    results to CSV and saves a plot of the learning curve.

    Args:
        estimator (object): Target classifier.
        scorer (object): Scoring function to be used.
        X_train (numpy.Array): Training features.
        y_train (numpy.Array): Training labels.
        data_name (str): Name of data set being tested.
        clf_name (str): Type of algorithm.
        cv (int): Number of folds in cross-validation splitting strategy.

    """
    # set training sizes and intervals
    train_sizes = np.arange(0.1, 1.0, 0.05)

    # set cross validation strategy to use StratifiedShuffleSplit
    cv_strategy = StratifiedShuffleSplit(n_splits=cv, random_state=0)

    # create learning curve object
    LC = learning_curve(estimator, X_train, y_train, cv=cv_strategy,
                        train_sizes=train_sizes, scoring=scorer, n_jobs=1)

    # extract training and test scores as data frames
    train_scores = pd.DataFrame(index=LC[0], data=LC[1])
    test_scores = pd.DataFrame(index=LC[0], data=LC[2])

    # save data frames to CSV
    resdir = 'results'
    res_tgt = '{}/{}'.format(resdir, clf_name)
    train_file = get_abspath('{}_LC_train.csv'.format(data_name), res_tgt)
    test_file = get_abspath('{}_LC_test.csv'.format(data_name), res_tgt)
    train_scores.to_csv(train_file, index=False)
    test_scores.to_csv(test_file, index=False)

    # create learning curve plot
    plt.figure(1)
    plt.plot(train_sizes, np.mean(train_scores, axis=1),
             marker='.', color='black', label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1),
             marker='.', color='blue', label='Cross-validation score')
    plt.legend(loc='best')
    plt.grid(linestyle='dotted')
    plt.xlabel('Samples used for training as a percentage of total')
    plt.ylabel('Balanced Accuracy')
    plt.ylim((0, 1.1))

    plt.title("Learning with {} on {}".format(clf_name, data_name))

    # save learning curve plot as PNG
    plotdir = 'plots'
    plot_tgt = '{}/{}'.format(plotdir, clf_name)
    plotpath = get_abspath('{}_LC.png'.format(data_name), plot_tgt)
    plt.savefig(plotpath)
    plt.close()


def create_timing_curve(
        estimator,
        dataset,
        data_name,
        clf_name
):
    """Generates a timing curve for the specified estimator, saves tabular
    results to CSV and saves a plot of the timing curve.

    Args:
        estimator (object): Target classifier.
        dataset(pandas.DataFrame): Source data set.
        data_name (str): Name of data set being tested.
        clf_name (str): Type of algorithm.

    """
    # set training sizes and intervals
    train_sizes = np.arange(0.1, 1.0, 0.05)

    # initialise variables
    train_time = []
    predict_time = []
    df_final = []

    # iterate through training sizes and capture training and predict times
    for i, train_data in enumerate(train_sizes):
        X_train, X_test, y_train, y_test = split_data(
            dataset, test_size=1 - train_data
        )
        start_train = timeit.default_timer()
        estimator.fit(X_train, y_train)
        end_train = timeit.default_timer()
        estimator.predict(X_test)
        end_predict = timeit.default_timer()
        train_time.append(end_train - start_train)
        predict_time.append(end_predict - end_train)
        df_final.append([train_data, train_time[i], predict_time[i]])

    # save timing results to CSV
    timedata = pd.DataFrame(
        data=df_final,
        columns=['Training Data Percentage', 'Train Time', 'Test Time'],
    )
    resdir = 'results'
    res_tgt = '{}/{}'.format(resdir, clf_name)
    timefile = get_abspath('{}_timing_curve.csv'.format(data_name), res_tgt)
    timedata.to_csv(timefile, index=False)

    # generate timing curve plot
    plt.figure(2)
    plt.plot(train_sizes, train_time, marker='.', color='b', label='Train')
    plt.plot(train_sizes, predict_time, marker='.', color='g', label='Predict')
    plt.legend(loc='best')
    plt.grid(linestyle='dotted')
    plt.xlabel('Samples used for training as a percentage of total')
    plt.ylabel('Elapsed user time in seconds')

    # save timing curve plot as PNG
    plotdir = 'plots'
    plt.title("Timing Curve with {} on {}".format(clf_name, data_name))
    plot_tgt = '{}/{}'.format(plotdir, clf_name)
    plotpath = get_abspath('{}_TC.png'.format(data_name), plot_tgt)
    plt.savefig(plotpath)
    plt.close()


def create_iteration_curve(
        estimator, X_train, X_test,
        y_train, y_test, data_name,
        clf_name, param, scorer
):
    """Generates an iteration curve for the specified estimator, saves tabular
    results to CSV and saves a plot of the iteration curve.

    Args:
        estimator (object): Target classifier.
        X_train (numpy.Array): Training features.
        X_test (numpy.Array): Test features.
        y_train (numpy.Array): Training labels.
        y_test (numpy.Array): Test labels.
        data_name (str): Name of data set being tested.
        clf_name (str): Type of algorithm.
        param (dict): Name of # iterations param for classifier.
        scorer (function): Scoring function.

    """
    # set variables
    iterations = np.arange(1, 300, 10)
    train_iter = []
    predict_iter = []
    final_df = []

    # start loop
    for i, iteration in enumerate(iterations):
        estimator.set_params(**{param: iteration})
        estimator.fit(X_train, y_train)
        train_iter.append(np.mean(cross_val_score(
            estimator, X_train, y_train, scoring=scorer, cv=5)))
        predict_iter.append(np.mean(cross_val_score(
            estimator, X_test, y_test, scoring=scorer, cv=5)))
        final_df.append([iteration, train_iter[i], predict_iter[i]])

    # save iteration results to CSV
    itercsv = pd.DataFrame(data=final_df, columns=[
        'Iterations', 'Train Accuracy', 'Test Accuracy'])
    resdir = 'results'
    res_tgt = '{}/{}'.format(resdir, clf_name)
    iterfile = get_abspath('{}_iterations.csv'.format(data_name), res_tgt)
    itercsv.to_csv(iterfile, index=False)

    # generate iteration curve plot
    plt.figure(3)
    plt.plot(iterations, train_iter, marker='.',
             color='b', label='Train Score')
    plt.plot(iterations, predict_iter, marker='.',
             color='g', label='Test Score')
    plt.legend(loc='best')
    plt.grid(linestyle='dotted')
    plt.xlabel('Number of iterations')
    plt.ylabel('Balanced Accuracy')
    plt.ylim((0, 1.1))

    plt.title("Interation with {} on {}".format(clf_name, data_name))
    # save iteration curve plot as PNG
    plotdir = 'plots'
    plot_tgt = '{}/{}'.format(plotdir, clf_name)
    plotpath = get_abspath('{}_IC.png'.format(data_name), plot_tgt)
    plt.savefig(plotpath)
    plt.close()


def create_validation_curve(
        estimator, X_train, y_train,
        data_name, clf_name,
        param_name, param_range, scorer
):
    """Generates an validation/complexity curve for the ANN estimator, saves
    tabular results to CSV and saves a plot of the validation curve.

    Args:
        estimator (object): Target classifier.
        X_train (numpy.Array): Training features.
        y_train (numpy.Array): Training labels.
        data_name (str): Name of data set being tested.
        clf_name (str): Type of algorithm.
        param_name (str): Name of parameter to be tested.
        param_range (numpy.Array): Range of parameter values to be tested.
        scorer (function): Scoring function.

    """
    # generate validation curve results
    train_scores, test_scores = validation_curve(
        estimator, X_train, y_train,
        param_name=param_name, param_range=param_range,
        cv=5, scoring=scorer, n_jobs=1
    )

    # generate validation curve plot
    plt.figure(4)
    if param_name is 'SVMR__gamma' or param_name is 'MLP__alpha':
        plt.semilogx(param_range, np.mean(train_scores, axis=1),
                     marker='.', color='b', label='Train Score')
        plt.semilogx(param_range, np.mean(test_scores, axis=1),
                     marker='.', color='g', label='Cross-validation Score')
    else:
        plt.plot(param_range, np.mean(train_scores, axis=1),
                 marker='.', color='b', label='Train Score')
        plt.plot(param_range, np.mean(test_scores, axis=1),
                 marker='.', color='g', label='Cross-validation Score')

    plt.legend(loc='best')
    plt.title("Validation Curve with {} on {}".format(clf_name, data_name))
    plt.grid(linestyle='dotted')
    plt.xlabel(param_name)
    plt.ylabel('Balanced Accuracy')
    plt.ylim((0, 1.1))

    # save iteration curve plot as PNG
    plotdir = 'plots'
    plot_tgt = '{}/{}'.format(plotdir, clf_name)
    plotpath = get_abspath('{}_VC.png'.format(data_name), plot_tgt)
    plt.savefig(plotpath)
    plt.close()


def create_decision_boundary(X, y, clf, data_name, clf_name):
    """Generates a decision boundary plot.
    Only suitable for two attributes and a class.

    Args:
        X (numpy.Array): Features.
        y (numpy.Array): Labels.
        clf (object): Target classifier.
        data_name (str): Name of data set being tested.
        clf_name (str): Type of algorithm.

    """
    cols = X.columns
    if len(cols) > 2:
        raise ("There are too many attributes to plot a decision boundary")

    # Plotting decision regions
    plot_decision_regions(np.array(X), np.array(y), clf=clf)

    # Adding axes annotations
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.grid(linestyle='dotted')
    plt.title("Decision Boundary with {} on {}".format(clf_name, data_name))

    # save iteration curve plot as PNG
    plotdir = 'plots'
    plot_tgt = '{}/{}'.format(plotdir, clf_name)
    plotpath = get_abspath('{}_DB.png'.format(data_name), plot_tgt)
    plt.savefig(plotpath)
    plt.close()


def create_decision_tree(dtree, data_name, clf_name):
    """Creates a tree chart for a decision tree classifier.

    Args:
        dtree (object): Decision Tree classifier.
        data_name (str): Name of data set being tested.
        clf_name (str): Type of algorithm.

    Returns:

    """
    dot_data = StringIO()
    export_graphviz(dtree.named_steps['DT'], out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    Image(graph.create_png())

    # save iteration curve plot as PNG
    plotdir = 'plots'
    plot_tgt = '{}/{}'.format(plotdir, clf_name)
    plotpath = get_abspath('{}_DT.png'.format(data_name), plot_tgt)
    graph.write_png(plotpath)


if __name__ == '__main__':

    # remove existing combined_results.csv file if it already exists
    combined_path = get_abspath('combined_results.csv', 'results')
    if os.path.exists(combined_path):
        os.remove(combined_path)

    # set scoring function
    scorer = make_scorer(balanced_accuracy)

    # get data names
    DATA_DIR = '../data/'
    data_files = os.listdir(DATA_DIR)
    dnames = [x.split('.')[0] for x in data_files]
    # dnames = ['abalone', 'banana', 'banknotes', 'bike_sharing_day', 'contraceptive', 'diabetes']

    dfs = {}
    for d in dnames:
        d_path = get_abspath('{}.csv'.format(d), 'data')
        dfs[d] = pd.read_csv(d_path)

    # instantiate dict of estimators
    estimators = {
        'KNN': None,
        'DT': None,
        'ANN': None,
        'SVM_RBF': None,
        'SVM_PLY': None,
        'Boosting': None
    }

    mnames = ['DT',
              'SVM_RBF',
              'SVM_PLY',
              'Boosting',
              'ANN',
              'KNN']

    # estimators with iteration param
    iterators = {
        'Boosting': 'ADA__n_estimators',
        'ANN': 'MLP__max_iter'
    }

    # validation curve parameter names and ranges
    vc_params = {
        'KNN': ('KNN__n_neighbors', np.arange(1, 40, 1)),
        'DT': ('DT__max_depth', np.arange(1, 50, 1)),
        'ANN': ('MLP__alpha', np.logspace(-10, 4, 20)),
        'SVM_RBF': ('SVMR__gamma', np.logspace(-9, 1, 15)),
        'SVM_PLY': ('SVMP__degree', np.arange(0.001, 10, 0.5)),
        'Boosting': ('ADA__learning_rate', np.arange(0.001, 10.0, 0.1)),
    }

    # start model evaluation loop
    # for df in dnames:
    for df_name, df in dfs.items():

        X_train, X_test, y_train, y_test = split_data(df)

        # load pickled models into estimators dict
        for m in mnames:
            mfile = '{}/{}_grid.pkl'.format(m, df_name)
            try:
                model = load_pickled_model(
                    get_abspath(mfile, filepath='models')
                )
                estimators[m] = model
            except IOError:
                pass

        # generate validation, learning, and timing curves
        for clf_name, estimator in estimators.items():
            if estimator is None:
                continue

            print('Evaluating: {}\nData: {}'.format(clf_name, df_name))
            start_time = timeit.default_timer()

            basic_results(
                estimator,
                X_test, y_test,
                data_name=df_name,
                clf_name=clf_name
            )

            create_learning_curve(
                estimator.best_estimator_,
                scorer,
                X_train, y_train,
                data_name=df_name,
                clf_name=clf_name
            )

            create_timing_curve(
                estimator.best_estimator_,
                dataset=df.copy(deep=True),
                data_name=df_name,
                clf_name=clf_name
            )

            create_validation_curve(
                estimator.best_estimator_,
                X_train, y_train,
                data_name=df_name,
                clf_name=clf_name,
                param_name=vc_params[clf_name][0],
                param_range=vc_params[clf_name][1],
                scorer=scorer
            )

            # two attributes and a class, draw decision boundary plot
            if len(df.columns) == 3:
                create_decision_boundary(
                    X=X_test,
                    y=y_test,
                    clf=estimator,
                    data_name=df_name,
                    clf_name=clf_name
                )

            if clf_name == 'DT':
                create_decision_tree(
                    dtree=estimator.best_estimator_,
                    data_name=df_name,
                    clf_name=clf_name
                )

            # generate iteration curves for ANN and AdaBoost classifiers
            if clf_name == 'ANN' or clf_name == 'Boosting':
                create_iteration_curve(
                    estimator.best_estimator_,
                    X_train, X_test, y_train, y_test,
                    data_name=df_name,
                    clf_name=clf_name,
                    param=iterators[clf_name],
                    scorer=scorer
                )

            end_time = timeit.default_timer()
            elapsed = end_time - start_time
            print('{} evaluated in {:f} seconds\n'.format(clf_name, elapsed))
