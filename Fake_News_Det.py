import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load the dataset
news_df = pd.read_csv('C:/Users/ASUS/Desktop/DataScienceInternship/news.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(news_df['text'], news_df['label'], test_size=0.2)

# Convert the text into a bag of words representation
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train a naive Bayes classifier on the training data
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Define the Streamlit app
def app():
    st.title("Fake News Detector")

    # Create a text input for the user to enter a news article
    article = st.text_input("Enter a news article:")

    if article:
        # Convert the article into a bag of words representation
        article_counts = vectorizer.transform([article])

        # Make a prediction using the trained classifier
        prediction = clf.predict(article_counts)[0]

        # Display the prediction to the user
        if prediction == 'FAKE':
            st.error("This article is fake.")
        else:
            st.success("This article is real.")

# Run the Streamlit app
if __name__ == '__main__':
    app()
