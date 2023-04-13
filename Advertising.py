import streamlit as st
import pandas as pd
import pickle

st.write("""
# Sales Prediction App

This app predicts the **Sales** according to advertising means!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0, 200, 147)
    Radio = st.sidebar.slider('Radio', 0, 200, 23)
    Newspaper = st.sidebar.slider('Newspaper', 0, 200, 30)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper,
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("Advertising_model.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)
