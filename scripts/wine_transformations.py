import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns
from scipy.stats import boxcox, yeojohnson, stats

"""
This module contains transformations for data in the 
wine dataset in order to normalize, transform and test data for feature extraction.
"""

def remove_outliers(maindata, feature, IQR, outlier_param):
    """
    Removes outliers from data for a given feature via Z-score or IQR methods.

    Args:
        maindata (pd.DataFrame) : A dataframe containing the feature.
        feature (str) : The feature name for outlier removal.
        IQR (bool) : Determines if the IQR is used for outlier removal or Z score. 
        outlier_param (int) : A parameter that gets used in the outlier calculations, either the scalar for IQR or Z score threshold.
    """
    data = maindata.copy()
    if IQR:
        quantile_1 = data[feature].quantile(0.25)
        quantile_3 = data[feature].quantile(0.75)
        IQR = quantile_3 - quantile_1
        lower_bound = quantile_1 - outlier_param * IQR
        upper_bound = quantile_3 + outlier_param * IQR
        data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]
        return data
    else: 
        feature_mean = data[feature].mean()
        feature_deviation = data[feature].std()
        data['Z_scores'] = data[feature].apply(func=(lambda x : (x - feature_mean)/feature_deviation))
        data = data[(data['Z_scores'] <= outlier_param) & (data['Z_scores'] >= (outlier_param*-1))]
        return data
    
def apply_transformations(maindata, feature, kind):
    """
    Applies transformation techniques to a feature in a dataframe, including reflective transformations.

    Args:
        maindata (pd.DataFrame) : A dataframe containing the feature.
        feature (str) : The feature name for outlier removal.    
        kind (str) : A method for transformation, either yeo, box, square, log
    Returns:
        data (pd.DataFrame) : Data with an applied normalization technique on a feature.
    """
    data = maindata.copy()
    if kind == "box":
        data[feature] = boxcox(data[feature])

    elif kind == "yeo":
        data[feature] = yeojohnson(data[feature])

    elif kind == "square":
        data[feature] = data[feature].apply(lambda x : np.square(maindata[feature].max() - x[feature]), axis=1)

    elif kind == "log":
        data[feature] = data[feature].apply(lambda x : np.log(maindata[feature].max() - x[feature]), axis=1)
    return data
    

def apply_normalization_techniques(maindata, feature, kind):
    """
    Applies normalization techniques to a feature in a dataframe.

    Args:
        maindata (pd.DataFrame) : A dataframe containing the feature.
        feature (str) : The feature name for outlier removal.    
        kind (str) : A method for normalization, either min-max, z-score   

    Returns:
        data (pd.DataFrame) : Data with an applied normalization technique on a feature.
    """
    data = maindata.copy()
    if kind == "min-max":
        maximum = data[feature].max()
        minimum = data[feature].min()
        data[feature] = (data[feature] - minimum) / (maximum - minimum)
    elif kind == "log":
        data[feature] = np.log(data[feature])
    elif kind == "inv-log":
        data[feature] = np.exp(data[feature])
    elif kind == "robust":
        median = data[feature].median()
        quantile_1 = data[feature].quantile(0.25)
        quantile_3 = data[feature].quantile(0.75)
        IQR = quantile_3 - quantile_1
        data[feature] = (data[feature] - median) / IQR
    elif kind == "z-score":
        mean = data[feature].mean()
        std = data[feature].std()
        data[feature] = (data[feature]-mean) / std
    return data
    
def sweetness(row):
    """
    Generates a new column called sweetness utilizing a weighted sum of correlated features.

    Args:
        row (pd.DataFrame) : A row from a pandas dataframe.

    Returns:
        calculation : The weighted sum returned for the new column.
    """

    return 2 * row['residual sugar'] + 0.5 * row['density'] + (10-row['alcohol'])


def acidity(row):
    """
    Generates a new column called acidity utilizing a weighted sum of correlated features.

    Args:
        row (pd.DataFrame) : A row from a pandas dataframe.

    Returns:
        calculation : The weighted sum returned for the new column.
    """

    return 4 * (1 - row['pH']/3.05) + 0.75 * row['chlorides'] +  4 * (row['fixed acidity'] - 9)

def crispness(row):
    """
    Generates a new column called crispness utilizing a weighted sum of correlated features.

    Args:
        row (pd.DataFrame) : A row from a pandas dataframe.

    Returns:
        calculation : The weighted sum returned for the new column.
    """
    
    return 4 * (row['fixed acidity']) + 2 * row['citric acid']

def fruitiness(row):
    """
    Generates a new column called fruitiness utilizing a weighted sum of correlated features.

    Args:
        row (pd.DataFrame) : A row from a pandas dataframe.

    Returns:
        calculation : The weighted sum returned for the new column.
    """
    free_so2 = row['free sulfur dioxide']
    pH = row['pH']
    molecular_so2 = free_so2 / (1 + 10**(pH-1.8))
    return  10 * (molecular_so2 - 0.6) + 2 * (1 - pH/3.05) + 2 * row['citric acid']

def map_affects(dataframes, feature, affects, transformation):
    """
    Plots the affects of data processing in regards to a set of dataframes and their changed features.

    Args:
        dataframes (lst) : A list containing dataframes.
        feature (str) : A feature that is being melted in the dataframes.
        affects (list(str)) : A list of strings of the affects that is being mapped via a dataframe melt.
        transformation (str) : The type of transformation that is being applied onto the data.
    Returns:
        melted_frame (pd.DataFrame) : A dataframe containing affects melted together for plotting. 
    """

    names = []
    for index in range(len(affects)):
        new_name = feature + " " + affects[index]
        dataframes[index] = dataframes[index].rename(columns={feature : new_name})
        names.append(new_name)

    combined_frame = pd.concat(dataframes, axis=0)
    data_melt = combined_frame.melt(id_vars='type', value_vars=names, var_name=transformation, value_name=feature)
    sns.displot(data_melt, hue='type', palette='rocket_r', x=feature, col=transformation)


