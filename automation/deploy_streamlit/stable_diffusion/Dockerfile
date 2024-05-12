FROM python:3.12-slim

WORKDIR /streamlit_app

RUN apt-get update && apt-get install -y \
  curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY image-generator-SDXL1.py image-generator-SDXL1.py


EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health