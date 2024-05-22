FROM rayproject/ray:2.12.0-gpu

WORKDIR /tmp

COPY requirements.txt /tmp/requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt

RUN sudo apt update && sudo apt install -y wget unzip && sudo apt-get clean
