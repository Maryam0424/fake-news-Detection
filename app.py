
# st.markdown("Developed by **Maryam Tariq**")


import joblib
import streamlit as st

# Load model and vectorizer
rf_model = joblib.load('./model/random_forest_model.pkl')
vectorization = joblib.load('./model/tfidf_vectorizer.pkl')

# Streamlit UI
st.title("Fake News Detector")
user_input = st.text_area("Enter a news article text:")

# Prediction Logic
if st.button("Predict"):
    if user_input.strip():
        # Transform the input text using the loaded vectorizer
        user_input_tfidf = vectorization.transform([user_input])  

        # Predict using the loaded model
        prediction = rf_model.predict(user_input_tfidf)

        # Display Prediction Result
        if prediction[0] == 1:
            st.success("Prediction: Real News")
        else:
            st.error("Prediction: Fake News")
    else:
        st.warning("Please enter some text to predict.")
