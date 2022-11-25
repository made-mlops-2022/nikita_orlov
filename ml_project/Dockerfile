FROM python:3.10

WORKDIR /ml_project
COPY req_docker.txt /ml_project/requirements.txt
RUN pip install -r requirements.txt

COPY app.py /ml_project/app.py
COPY src /ml_project/src
COPY models/model.pkl /ml_project/models/model.pkl
EXPOSE 80
CMD ["uvicorn", "app:predict_app", "--host", "0.0.0.0", "--port", "80"]