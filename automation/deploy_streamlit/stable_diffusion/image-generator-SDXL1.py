import io
import json
import base64
import requests
import os
import streamlit as st

endpoint = os.environ.get("ENDPOINT_URL")

def create_request_body(text, extras=None):
    if extras is None:
        extras = {}

    body = {
        "text_prompts": [{"text": text}],
    }
    return body

# 이미지 생성 요청 함수
def image_generate(endpoint, prompt):
    try:
        headers = {"Content-Type": "application/json"}
        request_body = create_request_body(
            text=prompt,
            extras={}
        )

        res = requests.post(endpoint, data=json.dumps(request_body), headers=headers)

        try:
            res_data = res.json()
        except json.JSONDecodeError:
            raise ValueError("Invalid response format from the endpoint")
        artifacts = res_data['output']['artifacts']
        if not artifacts:
            raise ValueError("No images returned from the endpoint")

        base64_img = artifacts[0].get("base64")
        bytes_data = base64.decodebytes(base64_img.encode())
        img_bytes = io.BytesIO(bytes_data)
        return img_bytes

    except Exception as e:
        print("Error:", e)
        return None

# Streamlit 앱
st.title("Image generate with your endpoint")

with st.form("form"):
    prompt = st.text_input("Description of the image to generate", disabled=not endpoint)
    submitted = st.form_submit_button("Generate", disabled=not endpoint)

if not endpoint:
    st.error('The endpoint url does not exist. Please try again later.', icon="🚨")

if submitted and prompt and endpoint:
    img_bytes = image_generate(endpoint, prompt)
    if img_bytes:
        st.image(img_bytes)
    else:
        st.error("Failed to generate image. Please try again.", icon="🚨")