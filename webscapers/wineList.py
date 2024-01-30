import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import numpy as np

def get_type(text):
    i=-1
    while text[i] != "/":
        i-=1
    return text[i+1:]

URL = "https://www.thebeerguy.ca/alphabetical-list-of-wine/page/"


reg_url = re.compile("\".*\"")
reg_type = re.compile("/\w*/")

wine_list = []
name_list = []
type_list = []


for page_num in range(1,15):
    temp_url = URL + str(page_num) + "/"
    page = requests.get(temp_url)
    soup = BeautifulSoup(page.content, "html.parser")

    text = soup.find_all("td")
    
    for element in text:
        name_list.append(element.text.strip())
        match = re.search(reg_url, str(element))
        new_text = match.group()
        matches = new_text.split("/")

        type_list.append(matches[2])

data1 = pd.DataFrame(name_list)
data2 = pd.DataFrame(type_list)
data1.to_csv('wine_name_list.csv', index=False)
data2.to_csv('wine_type_list.csv', index=False)





