import io
import json
import base64
import requests
import streamlit as st

def image_generate(endpoint, prompt):
  try:
    headers = {"Content-Type": "application/json"}
    body = {
      "prompt": prompt,
    }

    res = requests.post(endpoint, data=json.dumps(body), headers=headers)
    
    base64_img = json.loads(res.text)["images"][0]
    bytes_data = base64.decodebytes(base64_img.encode())
    img_bytes = io.BytesIO(bytes_data)
    return img_bytes

  except Exception as e:
    print("Error: ", e)

st.title("Image generate with your endpoint");

with st.sidebar:
    endpoint = st.text_input("Endpoint", placeholder="https://your-endpoint.com")

with st.form("form"):
  prompt = st.text_input("Description of the image to generate", disabled=not endpoint)
  submitted = st.form_submit_button("Generate")

if submitted and prompt and endpoint:
  st.image(image_generate(endpoint, prompt))
