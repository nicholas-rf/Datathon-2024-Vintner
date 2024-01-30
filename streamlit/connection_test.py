import streamlit as st
import pymysql

# Initialize connection.

def init_connection():
    return pymysql.connect(**st.secrets["singlestore"])

conn = init_connection()

def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

rows = run_query("SELECT * FROM wine_info;")
print(rows)
# for row in rows:
#     st.write(f"{row[0]} has a :{row[1]}:")
