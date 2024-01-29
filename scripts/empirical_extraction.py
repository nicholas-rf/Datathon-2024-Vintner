"""
This module contains functions that extract features from red wine empirically.
"""

def crispness(row):
    """
    Calculates crispness by utilizing free acidity.
    """
    free_acidity = row['fixed acidity']
    if free_acidity < 6:
        return 0
    elif 7 <= free_acidity < 9:
        return 1
    elif 9 <= free_acidity:
        return 2    

def sweetness(row):
    """
    Calculates the sweetness by utilizing the residual sugars, alcohol.
    """
    # should not consider density, is because it is dependant on sugar content
    res_sugar, alcohol = row['residual sugar'], row['alcohol']
    sweet_score = 0

    # Checks residual sugar
    if 5 <= res_sugar <= 11:
        sweet_score += 1
    elif 12 <= res_sugar <= 35:
        sweet_score += 2
    elif 35 < res_sugar:
        sweet_score += 3
    
    # Checks alcohol
    if alcohol < 10:
        sweet_score += 2
    elif 11 < alcohol < 12.5:
        sweet_score += 1
    
    return sweet_score
    
def vinegar(row):
    """
    Calculates the vinegar flavor.
    """
    volatile_acid = row['volatile acidity']

    if 1.4 <= volatile_acid:
        return 2
    elif 1.2 <= volatile_acid <= 1.3:
        return 1
    elif volatile_acid < 1.2:
        return 0
    
def acidity(row):
    """
    Calculates acid flavor.
    """
    # we can determine a direction and then a distance for this calculation
    pH = row['pH']
    average_acceptable_pH = 3.05 # predicated upon average recommended pH for red and white
    deviation_from_acceptable = pH/average_acceptable_pH
    distance = abs(pH - average_acceptable_pH)

    if deviation_from_acceptable > 1: # points towards basic
        if 1.0 > distance >= 0.5:
            return 1
        elif distance > 1.0:
            return 0
    elif deviation_from_acceptable == 1:
        return 2
    else:
        if 1.0 > distance >= 0.5:
            return 3
        elif distance >= 1.0:
            return 4

def fruitiness(row):
    """
    Calculates fruitiness.
    """
    total_so2 = row['total sulfur dioxide'] # no exact number for this so not including it
    
    free_so2 = row['free sulfur dioxide']
    pH = row['pH']
    molecular_so2 = free_so2 / (1 + 10**(pH-1.8)) # we need to turn it into molecular to determine its affect on aroma
    if 0.6 <= molecular_so2 <= 0.8:
        return 1
    elif 0.8 < molecular_so2:
        return 2
    else:
        return 0


