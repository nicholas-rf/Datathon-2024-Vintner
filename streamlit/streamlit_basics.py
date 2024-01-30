import streamlit as sl 
import pandas as pd

sl.title("Hello this is my streamlit app.")
sl.subheader("This is subheader.")
sl.text("this is basically p text.")
sl.markdown("**Hello** world")
sl.write("## H2")
sl.metric(label="Wind Speed", value="120ms\`-1")
sl.table(pd.DataFrame([[1,2],[3,4]]))
sl.image("220px-Red_Wine_Glass.jpg")
state = sl.checkbox("Checkbox", value=False)
if state:
    sl.write("You checked me.")

# Callback functions
def change():
    print(sl.session_state.checker)
sl.checkbox("Checkbox2", value=False, on_change=change, key="checker")

radio_btn = sl.radio("This is a question about wine.", options=(1,2,3,4,5))
print(radio_btn)

def btn_click():
    sl.write("Submitting")
btn=sl.button(label="Submit", on_click=btn_click)
multi = sl.multiselect("",options=(1,2,3,4,5))
sl.slider("Slider", min_value=1, max_value=5)