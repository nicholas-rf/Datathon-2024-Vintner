from openai import OpenAI

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
        {'role':'system', 'content':'You are an assistant that can read reviews of wine, create ratings for the users preferences in the categories of crispness, acidity, fruitiness and sweetness for the reviewer, while factoring in the grape and location.'},
        {'role':'system', 'content':'Your responses are always in the format of {"crispness": score , "acidity" : score, "fruitiness" : score, "sweetness" : score} and do not include any extra information.'},
        {'role':'system', 'content':'Your scores are out of 5. With 5 meaning that the user likes that quality and 0 meaning they do not.'},
        {'role':'system', 'content':'If the key word "EXPLAIN" is present in the prompt, you will explain your reasoning. If not, only respond with the assigned format.'},
        {'role':'system', 'content': "You take special care to understand the usage of adverbs like 'really', 'greatly' or others that are used to describe extremes."},
        {'role':'system', 'content': "If multiple reviews are provided consider all of them for the overall response"},
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
    return eval(completion.choices[0].message.content)

print(call_GPT())
