Vintner is a project done for the UCSB 2024 Datathon which focuses on utilizing the chemical properties of the red-wine-quality dataset to provide more scientific wine recommendations based on a users preferences.

This project requires the following libraries: streamlit, pandas, tensorflow, beautifulsoup4, requests, numpy, scikit-learn, sqlalchemy, OpenAI, and pymysql.

Please refer to ../SingleStore Notebooks/Report_Wine_Qualities.ipynb to read about the project more intensly.

All data that is used within files can bef found in the data folder.

The singlestore notebooks folder contains 2 notebooks, one containing the concrete exploratory data analysis in wine_transformations.ipynb and the othere contianing the full report on the project. 

Training the k-nn model is required to run the web app and can be done in the wine_recommender.py module. To run the web app, install streamlit and run it locally, changing the filepath for the trained k-nn model to its new location. If utilizing the SingleStore database, a secrets file must be made with the server login information present.

Lastly, the original red wine qualities dataset was concatenated with the white wine qualities dataset.