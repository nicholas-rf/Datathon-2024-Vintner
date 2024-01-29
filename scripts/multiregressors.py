from sklearn import linear_model, neighbors, model_selection, tree, multioutput, svm
import numpy as np
import pandas as pd
"""
Contains methods for multi-regression models in order to use alcohol content in other datasets for potential feature prediction
"""
# paramdict

# some inherent methods for multiple output already
# linear regression, knregressor, decision tree regressor, random forest regressor
def generate_splits(dataset):
    """
    Generates training splits of the dataset for the specified variables to use in regression 
    """




def multi_linear(features, targets):
    """
    Creates a multi-linear regression model.

    Args:
        training (pd.DataFrame) : The training split of the dataset.
        testing (pd.DataFrame) : The testing split of the dataset.
    Returns:
        multi_linear_model (sklearn.Model) : A trained linear model for multi-regression
    """
    model = linear_model.LinearRegression()

    model.fit(features, targets)


    # hyperparams for neighbors
    # n_neighbors: Int = 5,
    # *,
    # weights: ((...) -> Any) | Literal['uniform', 'distance'] | None = "uniform",
    # algorithm: Literal['auto', 'ball_tree', 'kd_tree', 'brute'] = "auto",
    # leaf_size: Int = 30,
    # p: Int = 2,
    # metric: str | ((...) -> Any) = "minkowski",
    # metric_params: dict | None = None,
    # n_jobs: Int | None = None

def multi_neighbors(n_neighbors, algorithm, leaf_size):
    """
    Creates a multi-neighbors regression model.

    Args:
        training (pd.DataFrame) : The training split of the dataset.
        testing (pd.DataFrame) : The testing split of the dataset.
    Returns:
        multi_linear_model (sklearn.Model) : A trained linear model for multi-regression
    """
    
    if algorithm not in ['ball_tree', 'kd_tree']:
        return neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, algorithm=algorithm)        
    else:
        return neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size)

def multi_tree(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, random_state=None):
    """
    Creates a multi decision tree regression model.

    Args:
        decision_tree args 
    Returns:
        multi_tree_model (sklearn.Model) : A trained multi decision tree model for regression
    """
    return tree.DecisionTreeRegressor(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes)

# maybe want to make the training and testing global? still considering structure

def direct_multiout_svm(epsilon, tol, C, loss, fit_intercept, intercept_scaling, dual, verbose):
    """
    Creates a multi output standard vector machine through use of the MultiOutputRegressor.

    Args:
        training (pd.DataFrame) : The training split of the dataset.
        testing (pd.DataFrame) : The testing split of the dataset.
    Returns:
        multi_linear_model (sklearn.Model) : A trained linear model for multi-regression   
    """
    model = svm.LinearSVR(epsilon=epsilon, tol=tol, C=C, loss=loss, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, dual=dual, verbose=verbose)
    wrapper = multioutput.MultiOutputRegressor(model)
    return wrapper


def evaluate_performance():
    """
    Evaluates models for a dataset with separate parameters and outputs success metrics to a csv.
    """ 
    # Think theres a seed in the regressors anyways
    np.random.seed(50)

    neighbor_grid = {
    'n_neighbors': [3, 5, 7, 10],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30, 40],
    }
    tree_grid = {
    'criterion': ['squared_error', 'friedman_mse', 'poisson'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'min_weight_fraction_leaf': [0.0, 0.1, 0.2],
    'max_features': [None, 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 20, 30],
    }
    svm_grid = model_selection.ParameterGrid({
    'epsilon': [0.0, 0.1, 0.2],
    'tol': [1e-5, 1e-4, 1e-3],
    'C': [0.1, 1.0, 10.0],
    'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
    'fit_intercept': [True, False],
    'intercept_scaling': [0.5, 1.0, 1.5],
    'dual': [True, False],
    'verbose': [0, 1, 2],
    })

    data = pd.read_csv('../DataThon/wine_data.csv')
    neighs = neighbors.KNeighborsRegressor()
    tree1 = tree.DecisionTreeRegressor()
    data.drop('Unnamed: 0', inplace=True, axis=1)
    predictors = [name for name in data.columns if name != "alcohol"]
    # clf = model_selection.GridSearchCV(neighs, neighbor_grid, scoring='neg_mean_squared_error', refit=True) 
    clf = model_selection.GridSearchCV(tree1, tree_grid, scoring='neg_mean_squared_error', refit=True)    
   
    clf.fit(data[['alcohol']], data[predictors])
    with open("../DataThon/test_all_results_musical.txt", 'w') as f:
        f.write(str(clf.cv_results_))
    
    with open("../DataThon/meepington_the_musical.txt", 'w') as f:
        f.write(str([clf.best_score_, clf.best_params_, clf.best_index_]))

if __name__ == "__main__":
    evaluate_performance()