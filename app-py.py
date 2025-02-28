import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd

# Vectorizer
vectorizer = open(r"c:\Users\motlalepula\Downloads/Vectorizer.pkl", "rb")
tweet_cv = joblib.load(vectorizer)  # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv(r"C:\Users\motlalepula\Desktop\app.py\train.csv")

# The main function where we will build the actual app
def main():
    """Tweet Classifier App with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("Tweet Classifier")
    st.subheader("Climate change tweet classification")

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
        st.info("Meet the Team")

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
                predictor = joblib.load(open(r"C:\Users\motlalepula\Desktop\app.py\Logistic_Regression.pkl", "rb"))
            elif selected_model == "Random Forest":
                predictor = joblib.load(open(r"C:\Users\motlalepula\Desktop\app.py\Decision_Tree_Classifier.pkl", "rb"))
            elif selected_model == "SVM":
                predictor = joblib.load(open(r"c:\Users\motlalepula\Downloads\Support_Vector.pkl", "rb"))
            else:
                st.error("Invalid model selection")

            prediction = predictor.predict(vect_text)

            # When the model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))

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
    st.subheader("Raw Twitter data and label")
    if st.checkbox('Show raw data'):  # data is hidden if the box is unchecked
        st.write(raw[['sentiment', 'message']])  # will write the df to the page

if __name__ == '__main__':
    main()
