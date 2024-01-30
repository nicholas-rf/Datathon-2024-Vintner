import streamlit as st
import pandas as pd 



st.title("Wine Recommender..")
red_white = st.radio("**1.** Red, White Or Both?",
                    options=("Red","White","Both"), captions=[":wine_glass: Red wine is typically richer and fuller-bodied, with flavors such as dark fruits, spices, and sometimes earthy or woody notes.",
                                                               ":champagne: White wine is generally lighter and more refreshing, with notes like citrus, apple, and pear, along with floral or mineral notes.",
                                                                ":partying_face: I like both/Don't Care"],
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
tartness = st.radio("**6.** On a scale of 1-5 how tart do you like your wine?",
                    options=(1,2,3,4,5), captions=["No tartness", "A little tart", "Some tart/Neutral", "Pretty tart", "Very tart"],
                    horizontal=True, index=None)

if st.button("Submit"):
    if None in [red_white, sweetness, crispness, fruitiness, tartness]:
        st.write("Not All Forms Have Been Completed")
    else:
        USER_PROFILE = (red_white.lower(), sweetness-1, acidity-1, crispness-1, fruitiness-1, tartness-1)
        st.write(str(USER_PROFILE))
        #st.switch_page("pages/results.py")

