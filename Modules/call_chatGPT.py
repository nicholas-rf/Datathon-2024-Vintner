from openai import OpenAI
import pandas as pd

"""
This module utilizes chatGPT to read in wine reviews from the Denmark Technical University wine-sensed dataset and determines what a users preferences are.
"""


# Establish the OpenAI() client
client = OpenAI()
sample_information = {"review" : ['Hmm, didnt enjoy how sweet this was, not going to drink it again', 'I really loved how acidic this wine was! Especially the hints of lemon and orange.'], "grape" : ['sangiovese', 'chardonnay'] ,"country" : ['italy', 'america']}

def call_GPT(information=sample_information):
    """
    Calls the chat gpt api in order to generate user preferences based off of their reviews.

    Args:
        information (dict) : A dictionary containing reviews, grape varieties and where the wine comes from to feed into the api call.
    
    Returns:
        response (str) : A string of a dictionary containing the ratings for the user based off of the review.
    """

    # Extract prompt info
    review_texts = information['review']
    grapes = information['grape']
    countries = information['country']
    
    # Create and send prompt
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {'role':'system', 'content':'You are an assistant that can read reviews of wine, create ratings for the users preferences in the categories of crispness, acidity, fruitiness sweetness and their preference for red or white for the reviewer, while factoring in the grape and location.'},
        {'role':'system', 'content':'Your responses are always in the format of {"crispness": score , "acidity" : score, "fruitiness" : score, "sweetness" : score, "preference" : color} and do not include any extra information.'},
        {'role':'system', 'content':'Your scores are out of 5. With 5 meaning that the user likes that quality and 0 meaning they do not.'},
        {'role':'system', 'content':'If the key word "EXPLAIN" is present in the prompt, you will explain your reasoning. If not, only respond with the assigned format.'},
        {'role':'system', 'content': "You take special care to understand the usage of adverbs like 'really', 'greatly' or others that are used to describe extremes."},
        {'role':'system', 'content': "If multiple reviews are provided consider all of them for the overall response"},
        {'role':'system', 'content': 'Only return results in the form of {"crispness": score , "acidity" : score, "fruitiness" : score, "sweetness" : score, "preference" : color}'},
        {"role": "user", "content":
        f"""
         Review1 : '{review_texts[0]}',
         Grape1 : {grapes[0]},
         Country1 : {countries[0]}, 
         Review2 : '{review_texts[1]}',
         Grape2 : {grapes[1]},
         Country2 : {countries[1]}, 
        """}])
    
    # Return the response
    response = completion.choices[0].message.content
    return response

def create_info(reviews, country, grapes):
    """
    Creates information about a user from a row in a data frame of wine reviews gathered from the DTU dataset.

    Args:
        reviews (list(str)) : A list of reviews.
        country (list(str)) : A list of countries.
        grapes (list(str)) : A list of grape names.

    Returns: 
        information (dict) : A dictionary containing information about a users wine tastes given their review. 
    """
    # Initialize a dictionary containing the relevant information from the reviews
    information = {
        "review" : [reviews[0], reviews[1]],
        "grape" : [grapes[0], grapes[1]],
        "country" : [country[0], country[1]]
        }
    
    # Return the information for use in the openAI api call
    return information

def create_user_DB():
    """
    Creates a csv containing users and their preferences of wine attributes.

    Args:
        None
    
    Returns:
        Downloads the dataframe into a csv at a filepath. 
    """
    
    # Read in the reviews that have been processed from the wine sensed dataset
    reviews = pd.read_csv('/Users/nick/Documents/GitHub/spingle-dingle/scripts/wine_reviews.csv')

    # Convert the first 500 reviews into a dictionary
    reviews = reviews.head(500)
    review_dict = reviews.to_dict()

    # Initialize the empty dataframe with columns for what a user likes
    df = pd.DataFrame(columns=['crispness', 'acidity', 'fruitiness', 'sweetness', 'preference'])

    # Initialize a counter to manage dataframe indexing during concatenation
    counter = 0

    # Run a for loop stepping by 2 to gather two reviews for 1 user
    for index in range(0, 500, 2):
        information = create_info([review_dict['review'][index], review_dict['review'][index+1]],
        [review_dict['country'][index], review_dict['country'][index+1]],
        [review_dict['grape'][index], review_dict['grape'][index+1]])
        response = call_GPT(information)
        if type(response) == str:
            try:
                df = pd.concat([df, pd.DataFrame(eval(response),  index=[counter])], axis=0)
                counter += 1
            except:
                pass

    # Ouput the results to a csv
    df.to_csv('/Users/nick/Documents/GitHub/spingle-dingle/scripts/wine_users.csv')


create_user_DB()

