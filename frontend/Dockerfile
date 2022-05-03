FROM python:3.8-slim

RUN mkdir /frontend

COPY requirements.txt /frontend

WORKDIR /frontend

RUN pip install -r requirements.txt

COPY . /frontend

EXPOSE 80

CMD ["streamlit", "run", "ui.py","--server.port", "80"]