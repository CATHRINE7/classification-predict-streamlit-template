import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
#from scipy import sparse
#from scipy.sparse import _csr
#import scipy.sparse
import pickle
#from sklearn.exceptions import InconsistentVersionWarning
#warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
#warnings.simplefilter("ignore", InconsistentVersionWarning)

# Vectorizer
vectorizer_path = "Vectorizer.pkl"
vectorizer_full_path = os.path.join(os.getcwd(), vectorizer_path)
with open(vectorizer_full_path, "rb") as vectorizer_file:
    tweet_cv = joblib.load(vectorizer_file)
# Load your raw data
raw = pd.read_csv("train.csv")

# Mapping for numeric predictions to labels
label_mapping = {-1: "Anti", 0: "Neutral", 1: "Pro", 2: "News"}

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Setting page configuration
    st.set_page_config(
        page_title="Tweet Classifier App",
        page_icon=":speech_balloon:",
        layout="wide",
    )

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f0f0; /* Light Gray */
            color: #333; /* Dark Gray */
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .st-sidebar {
            background-color: #3498db; /* Blue for Sidebar */
            color: #fff; /* White text in Sidebar */
        }
        .stSelectbox {
            color: #2ecc71; /* Green text in Selectbox */
        }
        .stTextInput, .stTextArea {
            background-color: #ffc0cb; /* Pink background for Text Input */
            color: #333; /* Dark Gray text in Text Input */
        }
        .stButton {
            background-color: #2ecc71; /* Green background for Button */
            color: #fff; /* White text on Button */
        }
        .stDataFrame th, .stDataFrame td {
            background-color: #3498db; /* Blue background for DataFrame headers */
            color: #fff; /* White text in DataFrame headers */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Your Streamlit app content goes here
    st.title("Tweet Classifier")
    st.subheader("Climate Change Tweet Classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Home", "Information", "Meet the Team", "Prediction", "Last Page"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Home":
        st.text("Welcome to the Tweet Classifier App!")

    elif selection == "Information":
        st.info("General Information")
        # You can read a markdown file from the supporting resources folder
        st.markdown("This app helps classify climate change-related tweets. Users can input a tweet, "
                    "and the app will predict the sentiment using machine learning models.")

    # Building out the "Meet the Team" page
    elif selection == "Meet the Team":
        st.success("Meet the Team")

        # Team Lead
        st.write("- Cathrine Mamosadi: Team Lead")
        st.write("  - Background: Leadership and Project Management")
        st.write("  - Experience: Led the overall project development and coordination of this project.")

        # Project Manager
        st.write("- Louretta Maluleke: Project Manager")
        st.write("  - Background: Project Management and Coordination")
        st.write("  - Experience: Managed project timelines, resources, and communication.")

        # Data Engineer
        st.write("- Kanya Nake: Data Engineer")
        st.write("  - Background: Data Engineering and Database Management")
        st.write("  - Experience: Responsible for Model training and engineering.")

        # Data Scientist
        st.write("- Apphiwe Maphumulo: Data Scientist")
        st.write("  - Background: Data Science and Machine Learning")
        st.write("  - Experience: Developed machine learning models and conducted data analysis.")

        # App Developer
        st.write("- Nino Tsolo: App Developer")
        st.write("  - Background:  Web Development")
        st.write("  - Experience: Developed the Streamlit web application for user interaction.")

    # Building out the "Prediction" page
    elif selection == "Prediction":
        st.info("Prediction with ML Models")

        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text", "Type Here")

        # Load your .pkl file with the model of your choice + make predictions
        model_options = ["Logistic Regression", "Random Forest", "SVM"]
        selected_model = st.selectbox("Choose Model", model_options)

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()

            if selected_model == "Logistic Regression":
                predictor = joblib.load(open(r"Logistic_Regression.pkl", "rb"))
            elif selected_model == "Random Forest":
                predictor = joblib.load(open(r"Decision_Tree_Classifier.pkl", "rb"))
            elif selected_model == "SVM":
                predictor = joblib.load(open(r"Support_Vector.pkl", "rb"))
            else:
                st.error("Invalid model selection")

            prediction_numeric = predictor.predict(vect_text)
            prediction_label = label_mapping.get(prediction_numeric[0], "Unknown")

            # When the model has successfully run, will print prediction
            st.success(f"Text Categorized as: {prediction_label}")

    # Building out the "Last Page" page
    elif selection == "Last Page":
        st.info("This is the Last Page")
        st.subheader("App Summary")
        st.write("Thank you for using the Tweet Classifier App! This app helps you classify climate change-related tweets.")

        # Next Steps
        st.subheader("Next Steps")
        st.write("Explore more features, provide feedback, or visit our website for related resources.")

        # Thank You Message
        st.subheader("Thank You!")
        st.write("Thank you for using the Tweet Classifier App. We appreciate your time")

    # Building out the "Raw Twitter data and label" section
    st.subheader("Raw Twitter Data and Label")
    if st.checkbox('Show Raw Data'):  # data is hidden if the box is unchecked
        st.dataframe(raw[['sentiment', 'message']])  # will write the df to the page

if __name__ == '__main__':
    main()
