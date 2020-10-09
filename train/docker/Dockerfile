FROM python:3.6-slim-stretch

WORKDIR /src

RUN pip3 install --upgrade pip setuptools && \
    python3 -m pip install numpy pandas tensorflow sklearn tensorflow_hub tensorflow_text tqdm pyyaml h5py

CMD ["python3", "trainingSentiment.py"]