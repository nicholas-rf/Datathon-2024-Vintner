import pandas as pd
import numpy as np

"""
This module contains methods for binning variables via their quantiles in order to assign numbers 0-4 for wines.
"""

def assign_quant_bin(dataframe, col, data_type=None):
    """
    Takes a dataset and uses numpy.quantile to create quantiles for binning.

    Args:
        dataframe (pd.DataFrame) : A dataframe containing the data for binning.
        col (str) : The column that bins are being created for.
        data_type (str) : Either red or white.
    
    Returns:
        quants (list(int)) : A list of quantiles for the given column.
    """

    if data_type:
        dataframe = dataframe[dataframe["type"]==data_type]
    quants =  [np.quantile(dataframe[col], i/10) for i in range(2,10,2)]
    return quants
    
def assign_even_bin(dataframe, col):
    cut_points = np.linspace(start=dataframe[col].min(), stop=dataframe[col].max(), num=5 + 1)
    labels = ['0', '1', '2', '3', '4']
    dataframe.loc[:, col] = pd.cut(dataframe[col], cut_points, labels=labels, include_lowest=True)
    return dataframe


def split_data(row, attribute, quants_red, quants_white, data_type=None):
    """
    Takes a row, an attribute, two sets of quantiles, and an optional wine color to creates bins
    based on the passed quantiles for the attribute. This function is meant to be applied to a data set through an .apply() method

    Args:
        row (pd.DataFrame) : Row of dataset to be evaluated.
        attribute (str) : A given attribute to be scored from 0-4.
        quants_red (list(int)) : A list of 4 numeric values associated with 0.2, 0.4, 0.6, 0.8, and 1.0 quantiles respectively.
        quants_white (list(int)) : similar to quants red only meant to evaluate white wines
        data_type (str) : An optional string signifying whether the quantiles is for red wine or white wine.

    Returns: 
        0-4 based on row quantile score

    Why use data_type? There are two reasons why I think you might want to use data_type, first if you have dataset that is only red or white wine and 
    there is no attribute for type then, you can use data_type to avoid a key error. Also say you do not want to give different quantile scores for red vs.
    white wines. Simply set data_type to 'red' and feel free to set quants_white to None and its always go down one path of the if else.
    """

    # Is the first value of quantiles the highest quantile for the data? 
    if (data_type == 'red') | (row["type"] =="red"):
        if row[attribute] < quants_red[0]:
            return 4
        elif quants_red[1] > row[attribute] >= quants_red[0]:
            return 3
        elif quants_red[2] > row[attribute] >= quants_red[1]:
            return 2
        elif quants_red[3] > row[attribute] >= quants_red[2]:
            return 1
        else:
            return 0
    elif  (data_type == "white") | (row["type"] == "white"):
        if row[attribute] < quants_white[0]:
            return 4
        elif quants_white[1] > row[attribute] >= quants_white[0]:
            return 3
        elif quants_white[2] > row[attribute] >= quants_white[1]:
            return 2
        elif quants_white[3] > row[attribute] >= quants_white[2]:
            return 1
        else:
            return 0
    

    ## General flow red_quants = assign_quant(wine, "pH", "red")
        # white_quants = assign_quant(wine, "pH", "white")
        # red_quants, white_quants
        # wine["new_col"] = wine.apply(split_data, axis=1, args=("pH",red_quants,white_quants))
        # wine.head()

    # for every feature that quantiles are being made for 
        # generate quantiles for red and white from that feature
        # apply the quantiles to the new column for that feature