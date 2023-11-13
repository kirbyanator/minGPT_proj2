FROM pytorch/pytorch:latest

RUN apt update && apt install -y git nano vim

COPY docker_requirements.txt /app/
WORKDIR /app
RUN pip install -r docker_requirements.txt

