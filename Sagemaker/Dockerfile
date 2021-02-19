FROM mcr.microsoft.com/azureml/o16n-sample-user-base/ubuntu-miniconda


RUN apt-get -y update && apt-get -y install openjdk-8-jdk && apt-get install -y --no-install-recommends \
         wget \
         python3 \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
#RUN apt-get install python3-distutils
# Need to trim down on the pip install here ; as a lot of packages are not needed 

RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && \
    pip install numpy scipy==1.1.0 modin[ray] pandas boto3==1.15.18 botocore==1.18.18 urllib3==1.21.1 numpy datetime pyarrow nltk xlrd flask gevent gunicorn && \
        rm -rf /root/.cache


ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# the folder needs to go into the docker image
COPY h2o_imdb /opt/program
WORKDIR /opt/program
RUN pip install -r requirements.txt
# this punkt might not be needed 
RUN python -m nltk.downloader punkt

RUN chmod +x /opt/program/serve
