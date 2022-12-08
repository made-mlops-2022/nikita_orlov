FROM base_image

COPY train_model.py /train_model.py
COPY utils.py /utils.py
COPY training_config.yaml /training_config.yaml

ENTRYPOINT ["python", "train_model.py"]