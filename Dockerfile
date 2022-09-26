# FROM ubuntu
# RUN apt-get update -y && \
#     apt-get install -y python-pip python-dev

# COPY src/requirements.txt /app/requirements.txt

# WORKDIR /app

# RUN pip install -r src/requirements 

# COPY . /app

# ENTRYPOINT ["python"]

# CMD ["app.py"]

FROM python:3.9.7-slim-buster

RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
RUN dvc repro

COPY . /app

CMD ["app.py"]
