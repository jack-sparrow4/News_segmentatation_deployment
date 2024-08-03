import streamlit as st
import joblib
from text_process import text_preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier


tfidf = joblib.load(r'models/tfidf.joblib', 'r')
rcf_model = joblib.load(r'models/rfc_model.joblib', 'r')
nb_model = joblib.load(r'models/nb_model.joblib', 'r')
# knn_model = joblib.load(r'models/knn_model.joblib', 'r')
dt_model = joblib.load(r'models/dt_model.joblib', 'r')

st.header("News classification")

news_classes = {
    "1" : "Technology", 
    "2" : "Business", 
    "3" : "Sports", 
    "4" : "Entertainment", 
    "5" : "Politics", 
    "-2": "Sorry, not enough data to classify"
}

with st.sidebar:
    st.write("Model selection")
    model = st.radio("Select one:", ['RandomForestClassifier', 'DecisionTreeClassifier', 'Multinomial Naive bayes']) #'KNN',

    st.subheader("About:")
    st.write("It's a News classification application. It will categorize news in 5 category.")
    st.write("1. Technology")
    st.write("2. Business")
    st.write("3. Sports")
    st.write("4. Entertainment")
    st.write("5. Politics")

input_text = st.text_input("Enter your News Article to classify.")


def model_load(model):
    if model == 'RandomForestClassifier':
        return rcf_model
    # elif model == 'KNN':
    #     return knn_model
    elif model == 'Multinomial Naive bayes':
        return nb_model
    elif model == 'DecisionTreeClassifier':
        return dt_model
    

def model_predict(model, input_text):
    model_obj = model_load(model)
    processed_text = text_preprocessing(input_text)
    text_vector = tfidf.transform(processed_text)
    output = model_obj.predict(text_vector)
    print("Model output", output)
    output_category = news_classes[str(output[0])]
    print(output_category)
    final_output = "The news category is : {}".format(output_category)
    return final_output

if st.button('Classify'):
    if input_text:
        category = model_predict(model, input_text)
        st.write(category)
    else:
        st.write("please enter the news article")
else:
    st.write("Please paste your news")
