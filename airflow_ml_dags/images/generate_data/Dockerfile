FROM base_image

COPY heart_cleveland_upload.csv /data.csv
COPY generate_data.py /generate_data.py

ENTRYPOINT ["python3", "generate_data.py"]