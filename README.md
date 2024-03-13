Vintner is the winner of the UCSB 2024 Datathon which focuses on utilizing the chemical properties of the red-wine-quality dataset to provide scientifically backed wine recommendations based on a users preferences. The main report containing all information can be found in 
'../SingleStore Notebooks/Report_Wine_Qualities.ipynb'.

This project requires the following libraries to be installed: streamlit, pandas, tensorflow, beautifulsoup4, requests, numpy, scikit-learn, sqlalchemy, OpenAI, and pymysql.

All data that is used within files can bef found in the data folder.

To run the web app:
Update the filepath in the streamlit_app_1.py to the locally stored version of the k-nn model called knnpickle_file in Data. Then move the .steamlit folder into the users folder, then cd into the streamlit folder, and call 'streamlit run streamlit_app_1.py' in the console.

Web app is currently not able to run as the single store database needs to be active.
