FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements_kubernetes.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
COPY kubernetes_app.py /app/app.py

CMD ["python3", "/app/app.py"]
