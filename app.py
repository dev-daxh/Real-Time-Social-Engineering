import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

# Ensure necessary NLTK data packages are downloaded
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model with error handling
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

try:
    tfidf = pickle.load(open(vectorizer_path, 'rb'))
    model = pickle.load(open(model_path, 'rb'))
except Exception as e:
    st.error("Error loading model files: " + str(e))
    st.stop()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. Predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


# Ignore this code 


# import streamlit as st
# import pickle

# # Load the model and vectorizer
# with open('/mnt/data/model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# with open('/mnt/data/vectorizer.pkl', 'rb') as vectorizer_file:
#     vectorizer = pickle.load(vectorizer_file)

# # Streamlit UI
# st.title("Spam Classification App")
# st.write("Enter a message below to check if it is Spam or Ham.")

# # User input
# user_input = st.text_area("Message", "")

# if st.button("Predict"):
#     if user_input:
#         # Transform input text using the loaded vectorizer
#         transformed_text = vectorizer.transform([user_input])
        
#         # Get the model's prediction
#         prediction = model.predict(transformed_text)[0]
        
#         # Display the result
#         prediction_label = "Spam" if prediction == 1 else "Ham"
#         st.subheader("Prediction:")
#         st.write(f"The message is classified as: **{prediction_label}**")
#     else:
#         st.write("Please enter a message to classify.")
