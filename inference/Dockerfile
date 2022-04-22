FROM python:3.8

WORKDIR /project

COPY main.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python" ]

CMD ["main.py" ]

