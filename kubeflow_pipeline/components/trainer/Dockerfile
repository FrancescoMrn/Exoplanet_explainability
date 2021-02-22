FROM google/cloud-sdk:latest

RUN apt-get update && \
    apt-get install --yes python3-pip

COPY . /code
WORKDIR /code

RUN pip install -U numpy xgboost scikit-learn google-cloud-storage