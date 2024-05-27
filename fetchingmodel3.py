import streamlit as st
import pickle as pk

# Load my trained model ^_^
with open('log_reg.pkl', 'rb') as f:
    log_reg = pk.load(f)

# Load vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pk.load(f)

# Function to preprocess input text and transform it into vector
def preprocess_text(text):
    return vectorizer.transform([text])

# Function to predict sentiment
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    prediction = log_reg.predict(preprocessed_text)
    return prediction[0]

# UI
def main():
    st.title('Tweet Sentiment Analysis')

    # Tweets Textbox
    tweet = st.text_area('Enter your tweet here:')

    if st.button('Predict Sentiment'):
        if tweet.strip() == '':
            st.warning('Please enter a tweet.')
        else:
            # Predict sentiment
            sentiment = predict_sentiment(tweet)
            if sentiment == 1:
                st.success('Positive Sentiment')
            else:
                st.error('Negative Sentiment')

if __name__ == '__main__':
    main()
