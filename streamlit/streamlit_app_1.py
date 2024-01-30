import streamlit as st
import pandas as pd 
import pymysql
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

def init_connection():
    return pymysql.connect(**st.secrets["singlestore"])

conn = init_connection()

def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

knn = pickle.load(open('/Users/nick/Documents/GitHub/spingle-dingle/data/knnpickle_file', 'rb'))

st.title("Vintner")
red_white = st.radio("**1.** Red or White",
                    options=(0,1), captions=[":wine_glass: Red wine is typically richer and fuller-bodied, with flavors such as dark fruits, spices, and sometimes earthy or woody notes.",
                                                               ":champagne: White wine is generally lighter and more refreshing, with notes like citrus, apple, and pear, along with floral or mineral notes."],
                    horizontal=True, index=None)
sweetness = st.radio("**2.** On a scale of 1-5 how sweet do you like your wine?",
                    options=(1,2,3,4,5), captions=["Dry", "Off Dry", "Medium Sweet", "Sweet", "All the Sweet"],
                    horizontal=True, index=None)
acidity = st.radio("**3.** On a scale of 1-5 how acidic do you like your wine?",
                    options=(1,2,3,4,5), captions=["Not very acidic/Flat", "Less Acidic", "About Average Acidity", "Quite acidic", "Very acidic"],
                    horizontal=True, index=None)
crispness = st.radio("**4.** On a scale of 1-5 how much do you like a crisp mouthfeel in your wine?",
                    options=(1,2,3,4,5), captions=["Very smooth, soft, or even creamy wine", "Not very crisp", "Medium/Neutral", "Somewhat crisp", "Very crisp, refreshing, maybe even rough"],
                    horizontal=True, index=None)
fruitiness = st.radio("**5.** On a scale of 1-5 how much do you enjoy fruity flavors in your wine?",
                    options=(1,2,3,4,5), captions=["No fruity notes", "A little fruity notes are good", "Fruity notes are okay/Neutral", "Quite fruity", "Very fruity"],
                    horizontal=True, index=None)
dryness = st.radio("**6.** On a scale of 1-5 how dry do you like your wine?",
                    options=(1,2,3,4,5), captions=["No dryness", "A little dry", "Some dry/Neutral", "Pretty dry", "Very dry"],
                    horizontal=True, index=None)
quality = st.radio("**6.** On a scale of 1-10 how high class is your taste?",
                    options=(1,2,3,4,5,6,7,8,9,10), captions=["Not very high class","Low class","Below average","Average","Above average","Moderately high class","Quite high class","Very high class","Extremely high class","The epitome of high class"],
                    horizontal=True, index=None)

if st.button("Submit"):
    if None in [red_white, quality, crispness, acidity, fruitiness, sweetness, dryness]:
        st.write("Not All Forms Have Been Completed, please submit all!")
        
    else:
        _, indices_to_select = knn.kneighbors([[red_white, quality, crispness, acidity, fruitiness, sweetness, 5-sweetness]])
        
        index_str = ', '.join(map(str, indices_to_select[0]))
        # data = pd.read_csv('/Users/nick/Documents/GitHub/spingle-dingle/data/final_wine_data.csv')
        rows = run_query(f"SELECT * FROM wine_info WHERE wine_index IN ({index_str});")
        unwrapped_list = [item for item in rows]

        print("The wines we recommend are the following!")
        
        for touple in unwrapped_list:
            st.write(f"We recommend {touple[4]}!")
