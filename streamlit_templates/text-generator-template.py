import time
import json
import requests
import streamlit as st

def text_generate(endpoint, prompt, max_token = 1024):
  try:
    headers = {"Content-Type": "application/json"}
    body = {
      "prompt": prompt,
      "temperature": 0,
      "topP": 1,
      "maxTokenCount": max_token,
      "stopSequences": [],
    }

    res = requests.post(endpoint, data=json.dumps(body), headers=headers)
    for word in json.loads(res.text)["results"][0]["outputText"].split():
      yield word + " "
      time.sleep(0.02)

  except Exception as e:
    print("Error: ", e)
    raise

st.title("Chatbot with your endpoint");

with st.sidebar:
  endpoint = st.text_input("Endpoint", placeholder="https://your-endpoint.com")
  max_token = st.slider("Max Token", min_value=64, max_value=4096, value=1024, step=64)

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

  with st.chat_message("assistant"):
    response = st.write_stream(text_generate(endpoint, prompt, max_token))
    st.session_state.messages.append({"role": "assistant", "content": response})
