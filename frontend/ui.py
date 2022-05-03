import requests
import json
import time
import streamlit as st


url = 'http://covpredict.ml:8000/api/predict/'
headers = {
    'accept': 'application/json'
}


uuid = st.text_input("User id: ")
audio_file = st.file_uploader("Audio file:")
gender = st.text_input("Gender: ")
age = st.number_input("Age: ")
cough_type = st.text_input("Cough type: ")
health_status = st.text_input("Health status: ")
note = st.text_input("Note: ")

if st.button('Submit'):
    metadata = json.dumps(
    {
        "uuid": uuid,
        "subject_gender" : gender,
        "subject_age" : age,
        "subject_cough_type": cough_type,
        "subject_health_status": health_status,
        "note": note
    })

    files = {
        'meta': (None, metadata),
        'audio_file': (audio_file.name, audio_file),
    }

    response = requests.post(url, headers=headers, files=files).json()
    st.write("Your assessment: ", response["assessment"])


    

