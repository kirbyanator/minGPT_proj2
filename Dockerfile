FROM pytorch/pytorch:latest

RUN apt update && apt install -y git nano vim

COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt

