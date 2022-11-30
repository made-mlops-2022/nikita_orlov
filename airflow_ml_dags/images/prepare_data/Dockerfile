FROM base_image

COPY prepare_data.py /prepare_data.py
COPY feature_params.yaml /feature_params.yaml

ENTRYPOINT ["python3", "prepare_data.py"]