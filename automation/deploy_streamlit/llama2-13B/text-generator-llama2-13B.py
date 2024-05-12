import os
import json
import requests
import streamlit as st

endpoint = os.environ.get("ENDPOINT_URL")

def text_generate(endpoint, prompt, max_gen_len=512):
    try:
        headers = {"Content-Type": "application/json"}
        body = {
            "prompt": prompt,
            "temperature": 0.5,
            "top_p": 0.9,
            "max_gen_len": max_gen_len,
        }

        res = requests.post(endpoint, data=json.dumps(body), headers=headers)
        data = json.loads(res.text)
        if 'output' in data:
            return data['output']
        else:
            return "No output returned by API"

    except Exception as e:
        st.error(f"Error: {e}")

st.title("Chatbot with your endpoint")

with st.sidebar:
    max_gen_len = st.slider("Max Generation Length", min_value=64, max_value=512, value=512, step=64)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Write your prompt here", disabled=not endpoint)

if endpoint and prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = text_generate(endpoint, prompt, max_gen_len)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
