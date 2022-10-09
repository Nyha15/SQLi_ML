import streamlit as st
from prediction import predict_probability

st.title("SQLi Detection")

query = st.text_input('Enter the SQL query:')

if st.button('Predict'):
    probability = predict_probability(query)
    predicted_class = np.argmax(probability)
    if predicted_class:
      st.markdown(f'### This is an SQL injection with probability: {probability[predicted_class]}')
    else:
      st.markdown(f'### This is not an SQL injection with probability: {probability[predicted_class]}')
