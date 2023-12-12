FROM ubuntu:latest

LABEL maintainer="Rohan"


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         nginx \
         ca-certificates \
         gcc \
         python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py
RUN pip install numpy scikit-learn==1.3.2 joblib modin pandas numpy datetime pyarrow boto3 nltk xlrd flask gevent gunicorn && \
        rm -rf /root/.cache


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY RandomForest /opt/program
RUN chown root:root /opt/program
USER root
RUN chmod 777 /opt/program/serve
WORKDIR /opt/program