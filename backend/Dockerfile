# Build a docker that run the service
# author: Xuan-Son Vu
# 26 Aug 2021

# base image
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

RUN mkdir /backend
# set the working directory
COPY requirements.txt /backend

WORKDIR /backend

RUN apt update && apt install ffmpeg libsndfile1 --yes --no-install-recommends && \
    pip install --upgrade pip && pip install -r requirements.txt

COPY . /backend

EXPOSE 8000
# start serving
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
